import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_handbook_pdf(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    c = canvas.Canvas(filepath, pagesize=letter)

    # --- PAGE 1: HR & Remote Work Policy ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 720, "GoLogic Engineering Handbook - Page 1: Work & Leave Policy")
    c.setFont("Helvetica", 11)
    text_p1 = [
        "1. Remote Work: Engineers may work remotely up to 3 days per week.",
        "Core collaboration hours are 10:00 AM to 3:00 PM AEST regardless of location.",
        "",
        "2. Annual Leave & Vacation: Full-time employees accrue 20 days of paid annual leave",
        "per year. Leave requests exceeding 5 consecutive days require Engineering Manager",
        "approval at least two weeks in advance.",
        "",
        "3. Home Office Stipend: Every new hire receives a one-time $1,200 AUD allowance",
        "to purchase an ergonomic chair, monitor, or mechanical keyboard.",
    ]
    y = 680
    for line in text_p1:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()

    # --- PAGE 2: Cloud & Security Architecture ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 720, "GoLogic Engineering Handbook - Page 2: Security & Infrastructure")
    c.setFont("Helvetica", 11)
    text_p2 = [
        "4. Secret Management: Production database credentials are managed via HashiCorp Vault.",
        "All database passwords and API keys automatically rotate every 90 days.",
        "Hardcoding secrets in Git repositories triggers an immediate CI/CD pipeline block.",
        "",
        "5. Deployment Window: Production deployments to AWS EKS are permitted Monday through",
        "Thursday between 9:00 AM and 4:00 PM. Friday deployments are strictly prohibited",
        "unless patching a critical CVE security vulnerability with CTO sign-off.",
    ]
    y = 680
    for line in text_p2:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()

    # --- PAGE 3: Incident Response & On-Call ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 720, "GoLogic Engineering Handbook - Page 3: Incident Response")
    c.setFont("Helvetica", 11)
    text_p3 = [
        "6. Severity 1 (SEV-1) Outages: A SEV-1 incident is defined as complete unavailability",
        "of the core payment API affecting more than 5% of active users.",
        "The on-call engineer must acknowledge PagerDuty alerts within 15 minutes.",
        "",
        "7. Post-Mortems: A blameless post-mortem document must be published in Confluence",
        "within 48 hours of resolving any SEV-1 or SEV-2 incident.",
    ]
    y = 680
    for line in text_p3:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()

    c.save()
    print(f"✅ Created 3-page sample PDF at: {filepath}")


if __name__ == "__main__":
    create_handbook_pdf("sample_data/handbook.pdf")
