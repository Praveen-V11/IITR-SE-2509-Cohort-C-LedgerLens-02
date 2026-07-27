"""
LedgerLens front end.

Deliberately thin - all the real logic lives behind the FastAPI service.
This just gives a human somewhere to drop a photo and somewhere to clear
the review queue, which is exactly what the brief asks for.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("LEDGERLENS_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="LedgerLens", page_icon="\U0001f9fe", layout="wide")
st.title("LedgerLens")
st.caption("Confidence-aware document intelligence for receipts and invoices")

tab_upload, tab_review = st.tabs(["Upload", "Review queue"])

with tab_upload:
    st.subheader("Drop a receipt or invoice")
    uploaded = st.file_uploader("Image file", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            #st.image(uploaded, caption=uploaded.name, use_container_width=True)
            try:
                st.image(uploaded, caption=uploaded.name, use_container_width=True)
            except TypeError:
                st.image(uploaded, caption=uploaded.name, use_column_width=True)

        if st.button("Extract", type="primary"):
            with st.spinner("Running moderation + extraction..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                try:
                    resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=60)
                except requests.exceptions.ConnectionError:
                    st.error(f"Couldn't reach the API at {API_BASE}. Is it running?")
                    resp = None

                if resp is not None:
                    with col_result:
                        if resp.status_code == 422:
                            reason = resp.json().get("detail", {}).get("blocked_reason", "unspecified")
                            st.error(f"Blocked by moderation: {reason}")
                        elif resp.status_code != 200:
                            st.error(f"Extraction failed: {resp.text}")
                        else:
                            data = resp.json()
                            status = data["status"]
                            badge = "\U0001f7e2 auto-approved" if status == "auto_approved" else "\U0001f7e1 needs review"
                            st.markdown(f"**Status:** {badge}")
                            st.metric("Overall confidence", f"{data['overall_confidence']:.0%}")
                            if data["flagged_fields"]:
                                st.write("Flagged fields:")
                                st.write(", ".join(data["flagged_fields"]))
                            st.json(data)

with tab_review:
    st.subheader("Pending review")
    if st.button("Refresh queue"):
        st.rerun()

    try:
        queue_resp = requests.get(f"{API_BASE}/review", timeout=30)
        queue = queue_resp.json() if queue_resp.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        st.error(f"Couldn't reach the API at {API_BASE}. Is it running?")
        queue = []

    if not queue:
        st.info("Nothing waiting on review right now.")
    else:
        for item in queue:
            with st.expander(f"{item['filename']} \u2014 flagged: {', '.join(item['flagged_fields'])}"):
                extracted = item["extracted"]

                header_df = pd.DataFrame(
                    [
                        {"field": "vendor_name", "value": extracted["vendor_name"], "confidence": extracted["vendor_confidence"]},
                        {"field": "invoice_number", "value": extracted.get("invoice_number") or "", "confidence": extracted["invoice_number_confidence"]},
                        {"field": "document_date", "value": extracted.get("document_date") or "", "confidence": extracted["date_confidence"]},
                        {"field": "subtotal", "value": str(extracted["subtotal"]), "confidence": extracted["subtotal_confidence"]},
                        {"field": "tax", "value": str(extracted["tax"]), "confidence": extracted["tax_confidence"]},
                        {"field": "total", "value": str(extracted["total"]), "confidence": extracted["total_confidence"]},
                    ]
                )

                edited = st.data_editor(
                    header_df,
                    key=f"editor_{item['document_id']}",
                    column_config={
                        "confidence": st.column_config.ProgressColumn(
                            "confidence", min_value=0.0, max_value=1.0, format="%.2f"
                        )
                    },
                    disabled=["field", "confidence"],
                    hide_index=True,
                    use_container_width=True,
                )

                if extracted["line_items"]:
                    st.caption("Line items")
                    st.dataframe(pd.DataFrame(extracted["line_items"]), use_container_width=True, hide_index=True)

                reviewer = st.text_input("Reviewer name", value="", key=f"reviewer_{item['document_id']}")
                if st.button("Approve", key=f"approve_{item['document_id']}"):
                    corrections = {row["field"]: row["value"] for _, row in edited.iterrows()}
                    payload = {
                        "document_id": item["document_id"],
                        "corrected_fields": corrections,
                        "reviewer": reviewer or "unknown",
                    }
                    resp = requests.post(f"{API_BASE}/approve", json=payload, timeout=30)
                    if resp.status_code == 200:
                        st.success("Approved.")
                        st.rerun()
                    else:
                        st.error(f"Approval failed: {resp.text}")
