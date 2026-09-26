"""
PDF Work Order Generator for IIT Guwahati Estates Office
Generates professional maintenance work orders for road repairs and full segment resurfacing.
"""

import os
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_work_order_pdf(
    work_order_data: Dict[str, Any],
    cluster_items: List[Dict[str, Any]],
    output_pdf_path: str = "IITG_Estates_Work_Order.pdf"
) -> str:
    """
    Generates a formal IIT Guwahati Estates Office Road Maintenance Work Order PDF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#475569'),
        alignment=1
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    elements = []

    # Header
    elements.append(Paragraph("INDIAN INSTITUTE OF TECHNOLOGY GUWAHATI", title_style))
    elements.append(Paragraph("ESTATES AND WORKS SECTION • ROADS & CIVIL INFRASTRUCTURE DIVISION", subtitle_style))
    elements.append(Paragraph("Predictive Pavement Management System — Official Work Order", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

    # Meta Table
    wo_id = work_order_data.get("work_order_id", f"WO-IITG-{datetime.now().strftime('%Y%m%d')}-01")
    date_str = datetime.now().strftime("%d %B %Y, %H:%M IST")
    action_type = work_order_data.get("action", "Full Segment Resurfacing")
    route_name = work_order_data.get("route_zone", "Core_V_North_Loop").replace("_", " ")

    meta_table_data = [
        [
            Paragraph(f"<b>Work Order ID:</b> {wo_id}", body_style),
            Paragraph(f"<b>Date Issued:</b> {date_str}", body_style)
        ],
        [
            Paragraph(f"<b>Target Route:</b> {route_name}", body_style),
            Paragraph(f"<b>Action Type:</b> <font color='#2563EB'><b>{action_type}</b></font>", body_style)
        ],
        [
            Paragraph(f"<b>Center GPS:</b> {work_order_data.get('center_gps', '26.19120, 91.69250')}", body_style),
            Paragraph(f"<b>Defect Cluster Count:</b> {work_order_data.get('defect_count', 4)} detected defects", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # Predictive Engineering Assessment Section
    elements.append(Paragraph("1. PREDICTIVE ENGINEERING & RISK ASSESSMENT", section_heading))
    avg_rdi = work_order_data.get("mean_rdi", 0.62)
    avg_pdi = work_order_data.get("mean_pdi", 1.25)
    area_sqm = work_order_data.get("total_area_sqm", 6.8)
    
    eng_text = (
        f"This 100-meter corridor was clustered via Spatial DBSCAN combining multi-modal inputs "
        f"(proactive CCTV traffic telemetry, 100 Hz bicycle accelerometer vertical shocks & crowdsourced reports). "
        f"The aggregate <b>Road Damage Index (RDI) is {avg_rdi:.3f}</b> with a cumulative surface defect area of <b>{area_sqm:.2f} m²</b>. "
        f"Compounded by pavement aging, the <b>Predictive Degradation Index (PDI) has reached {avg_pdi:.3f}</b>, "
        f"crossing the critical threshold for sub-base deterioration. Reactive point patching is hereby discontinued "
        f"for this segment in favor of predictive 100m cold-milling and asphaltic concrete resurfacing."
    )
    elements.append(Paragraph(eng_text, body_style))
    elements.append(Spacer(1, 12))

    # Financial Cost-Benefit Comparison Section
    elements.append(Paragraph("2. LIFECYCLE COST SAVINGS ANALYSIS (GRANICA COMPLIANT)", section_heading))
    spot_cost = work_order_data.get("spot_repair_cost_inr", 30000)
    resurface_cost = work_order_data.get("full_resurface_cost_inr", 34000)
    lifecycle_savings = work_order_data.get("projected_cost_savings_pct", 54.7)

    cost_table_data = [
        [Paragraph("<b>Metric / Parameter</b>", body_style), Paragraph("<b>Reactive Spot Patching</b>", body_style), Paragraph("<b>Predictive Full Resurfacing</b>", body_style)],
        [Paragraph("Immediate Repair Cost", body_style), Paragraph(f"Rs. {spot_cost:,.2f}", body_style), Paragraph(f"Rs. {resurface_cost:,.2f}", body_style)],
        [Paragraph("Expected Longevity", body_style), Paragraph("6 - 9 Months (High monsoon failure)", body_style), Paragraph("4 - 5 Years (Durable wear course)", body_style)],
        [Paragraph("Projected 3-Year Maintenance", body_style), Paragraph(f"Rs. {int(spot_cost * 2.5):,.2f}", body_style), Paragraph(f"Rs. {resurface_cost:,.2f}", body_style)],
        [Paragraph("<b>Net Campus Savings</b>", body_style), Paragraph("Baseline (0% Savings)", body_style), Paragraph(f"<b><font color='#059669'>{lifecycle_savings}% Cost Reduction</font></b>", body_style)],
    ]
    cost_table = Table(cost_table_data, colWidths=[180, 180, 180])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94A3B8')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(cost_table)
    elements.append(Spacer(1, 14))

    # Defect Manifest Table
    elements.append(Paragraph("3. SEGMENT DEFECT MANIFEST", section_heading))
    manifest_headers = [
        Paragraph("<b>Observation ID</b>", body_style),
        Paragraph("<b>Source</b>", body_style),
        Paragraph("<b>Area (m²)</b>", body_style),
        Paragraph("<b>Peak Shock</b>", body_style),
        Paragraph("<b>PDI Score</b>", body_style)
    ]
    manifest_rows = [manifest_headers]
    
    for item in cluster_items[:6]: # Limit to top items to fit neatly on 1 page
        manifest_rows.append([
            Paragraph(str(item.get("observation_id", "OBS_001")), body_style),
            Paragraph(str(item.get("data_source", "CCTV")), body_style),
            Paragraph(f"{item.get('pothole_area_sqm', 0.5):.2f}", body_style),
            Paragraph(f"{item.get('z_accel_peak_g', 1.9):.2f}g", body_style),
            Paragraph(f"{item.get('predictive_degradation_score', 0.9):.3f}", body_style)
        ])

    manifest_table = Table(manifest_rows, colWidths=[130, 110, 100, 100, 100])
    manifest_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(manifest_table)
    elements.append(Spacer(1, 18))

    # Sign-off section
    elements.append(Paragraph("4. AUTHORIZATION & SIGN-OFF", section_heading))
    sign_table_data = [
        [
            Paragraph("<b>Recommended By:</b><br/>Lead Data Scientist / AI Vision Engine<br/>IIT Guwahati Estates Automation", body_style),
            Paragraph("<b>Approved By:</b><br/>Executive Engineer (Civil)<br/>Estates & Works Section, IIT Guwahati", body_style)
        ]
    ]
    sign_table = Table(sign_table_data, colWidths=[270, 270])
    sign_table.setStyle(TableStyle([
        ('LINEBEFORE', (1,0), (1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(sign_table)

    doc.build(elements)
    return output_pdf_path

if __name__ == "__main__":
    sample_wo = {
        "work_order_id": "WO-IITG-EST-2026-0042",
        "route_zone": "Core_V_North_Loop",
        "center_gps": "26.19120, 91.69250",
        "action": "Full Segment Resurfacing",
        "defect_count": 5,
        "mean_rdi": 0.684,
        "mean_pdi": 1.482,
        "total_area_sqm": 8.45,
        "spot_repair_cost_inr": 37500,
        "full_resurface_cost_inr": 34000,
        "projected_cost_savings_pct": 63.7
    }
    sample_items = [
        {"observation_id": "OBS_IITG_0001", "data_source": "MOBILE_RUN", "pothole_area_sqm": 1.84, "z_accel_peak_g": 3.42, "predictive_degradation_score": 1.54},
        {"observation_id": "OBS_IITG_0009", "data_source": "CCTV", "pothole_area_sqm": 2.10, "z_accel_peak_g": 2.10, "predictive_degradation_score": 1.41},
        {"observation_id": "OBS_IITG_0017", "data_source": "STUDENT_APP", "pothole_area_sqm": 1.65, "z_accel_peak_g": 1.00, "predictive_degradation_score": 1.28},
        {"observation_id": "OBS_IITG_0025", "data_source": "MOBILE_RUN", "pothole_area_sqm": 2.86, "z_accel_peak_g": 3.88, "predictive_degradation_score": 1.70},
    ]
    out_path = generate_work_order_pdf(sample_wo, sample_items, "sample_work_order.pdf")
    print("Generated sample PDF work order:", out_path)
