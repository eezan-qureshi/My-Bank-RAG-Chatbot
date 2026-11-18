from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path("data") / "my_bank_handbook.pdf"

PAGES = [
    # ---------------------------------------------------------------- page 1
    [
        ("My Bank Customer Handbook", None),
        ("(Sample document with fictional details for a classroom project.)", None),
        ("About My Bank", [
            "My Bank is a fictional bank used for a RAG chatbot demo.",
            "The helpline 0800-00000 is open 24 hours a day, 7 days a week.",
            "The support email is support@mybank.example.",
        ]),
        ("Account Types", [
            "My Bank Basic Account: the minimum opening deposit is PKR 1,000. There is no monthly maintenance fee and no minimum balance.",
            "My Bank Savings Account: the minimum opening deposit is PKR 5,000. Profit is credited to the account every month.",
            "My Bank Current Account: the minimum opening deposit is PKR 25,000. A monthly fee of PKR 500 applies if the average balance falls below PKR 25,000.",
        ]),
        ("Documents Required to Open an Account", [
            "Bring your original CNIC and one passport-size photograph.",
            "You also need proof of income or proof of address, such as a recent utility bill.",
            "Business customers must also bring their NTN certificate.",
            "New accounts are normally activated within 2 working days after the documents are verified.",
        ]),
    ],
    # ---------------------------------------------------------------- page 2
    [
        ("Debit Cards and ATMs", [
            "A new debit card is delivered within 7 working days after the account is opened.",
            "The annual fee is PKR 1,200 for the Classic card and PKR 3,000 for the Platinum card.",
            "The daily ATM withdrawal limit is PKR 50,000 for the Classic card and PKR 100,000 for the Platinum card.",
            "Withdrawals from My Bank ATMs are free. Withdrawals from another bank's ATM cost PKR 25 per transaction.",
            "The daily limit for online card purchases is PKR 100,000.",
        ]),
        ("Lost or Stolen Card", [
            "If your card is lost, stolen or you see a transaction you do not recognise, block the card immediately in the My Bank Mobile app: open Cards, then tap Block Card.",
            "Then call the helpline 0800-00000, which is available 24/7.",
            "A replacement card costs PKR 500 and is delivered within 5 working days.",
        ]),
        ("Changing Your Card PIN", [
            "You can change your card PIN at any My Bank ATM or inside the My Bank Mobile app under Cards, then Change PIN.",
            "Never share your PIN with anyone, including bank staff.",
        ]),
    ],
    # ---------------------------------------------------------------- page 3
    [
        ("My Bank Mobile App", [
            "To register, download the My Bank Mobile app and tap Register.",
            "Enter your account number and mobile number, then enter the OTP sent to you by SMS.",
            "Finally, create a 6-digit passcode for the app.",
            "If you forget your passcode, tap Forgot Passcode and verify yourself with an OTP, or visit any branch.",
        ]),
        ("Fund Transfers", [
            "Transfers between My Bank accounts are free and instant, with a daily limit of PKR 500,000.",
            "Transfers to other banks (IBFT) cost PKR 10 per transaction and normally arrive within 2 minutes. The daily limit for other-bank transfers is PKR 200,000.",
            "Utility bill payments in the app are free.",
        ]),
        ("Statements", [
            "An e-statement is emailed every month and is free.",
            "A paper statement costs PKR 200 per statement.",
        ]),
        ("Complaints", [
            "You can lodge a complaint through the app, at any branch, or by email at complaints@mybank.example.",
            "Every complaint is resolved within 10 working days.",
        ]),
    ],
    # ---------------------------------------------------------------- page 4
    [
        ("My Bank Personal Loan", [
            "The loan amount ranges from PKR 50,000 to PKR 3,000,000, with a repayment period of 12 to 60 months.",
            "Applicants must be between 22 and 60 years old. Salaried applicants need at least 6 months in their current job.",
            "The current markup rate is shown in the loan calculator inside the My Bank Mobile app.",
        ]),
        ("Security Rules", [
            "My Bank staff will never ask for your PIN, OTP, CVV, password or full card number by phone, SMS, WhatsApp or email.",
            "If someone asks for these details, hang up and call the helpline 0800-00000.",
            "Never click on links in unknown SMS messages that claim to be from the bank.",
        ]),
        ("Branch Timings", [
            "Branches are open Monday to Friday from 9 am to 5 pm, and on Saturday from 9 am to 1 pm. Branches are closed on Sunday.",
            "ATMs and the mobile app work 24 hours a day.",
        ]),
        ("Dormant Accounts", [
            "If an account has no customer transaction for 12 months, it becomes dormant.",
            "To reactivate it, visit a branch with your original CNIC.",
        ]),
    ],
]


def build():
    OUT.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=22, spaceAfter=10)
    h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=11, leading=15)

    story = []
    for i, page in enumerate(PAGES):
        for heading, lines in page:
            if lines is None:                       # title / subtitle lines
                story.append(Paragraph(heading, title if i == 0 and heading.startswith("My Bank") else body))
                continue
            story.append(Paragraph(heading, h))
            story.append(Paragraph(" ".join(lines).replace("&", "&amp;"), body))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm, title="My Bank Handbook").build(story)
    print("Created", OUT)


if __name__ == "__main__":
    build()
