"""One-off generator for the customer_service preset FAQ PDF.

Run from the repo root:  python backend/presets/generate.py
Requires fpdf2:          pip install fpdf2
Writes backend/presets/customer-faq.pdf
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).parent / "customer-faq.pdf"

SECTIONS = [
    (
        "About Acme Widgets",
        [
            (
                "What does Acme Widgets sell?",
                "Acme Widgets sells widgets and gadget accessories, including the classic widget, "
                "the widget pro, and a range of mounts, cases, and charging accessories.",
            ),
            (
                "Where are you located?",
                "Our headquarters is in Springfield. We ship to customers worldwide from our "
                "distribution centers in the United States and Europe.",
            ),
        ],
    ),
    (
        "Orders",
        [
            (
                "How do I place an order?",
                "Place an order on our website by adding items to your cart and checking out. "
                "You will receive a confirmation email right away.",
            ),
            (
                "Can I change or cancel my order?",
                "You can change or cancel an order within two hours of placing it by contacting "
                "support. After that, the order is already being prepared.",
            ),
            (
                "How do I check my order status?",
                "Use the tracking link in your confirmation email, or ask us here with your order "
                "number and we can look it up for you.",
            ),
        ],
    ),
    (
        "Shipping",
        [
            (
                "How long does shipping take?",
                "Standard shipping takes three to five business days in the United States and five "
                "to ten business days internationally. Express shipping takes one to two business "
                "days.",
            ),
            (
                "How much does shipping cost?",
                "Shipping is free on orders over fifty dollars. Otherwise, standard shipping is "
                "four ninety-nine and express is twelve ninety-nine.",
            ),
            (
                "Do you ship internationally?",
                "Yes, we ship to most countries. International shipping is available at checkout, "
                "and duties may apply depending on your location.",
            ),
        ],
    ),
    (
        "Returns and Refunds",
        [
            (
                "What is your return policy?",
                "You can return most items within thirty days of delivery for a full refund. Items "
                "must be unused and in their original packaging.",
            ),
            (
                "How do I start a return?",
                "Contact support with your order number and we will email you a prepaid return "
                "label. Drop the package at any carrier location and you are done.",
            ),
            (
                "When will I get my refund?",
                "Refunds are issued within five business days after we receive your return. It can "
                "take up to ten days for the refund to appear on your statement.",
            ),
        ],
    ),
    (
        "Warranty",
        [
            (
                "Does my product come with a warranty?",
                "Yes, every widget comes with a twelve month warranty covering manufacturing "
                "defects. Damage from misuse or accidents is not covered.",
            ),
            (
                "How do I make a warranty claim?",
                "Contact support with your order number and a short description of the problem. We "
                "will send a replacement or repair within a few business days.",
            ),
        ],
    ),
    (
        "Payments",
        [
            (
                "What payment methods do you accept?",
                "We accept all major credit cards, PayPal, and Apple Pay. All payments are "
                "processed securely.",
            ),
            (
                "Is my payment information safe?",
                "Yes. We use encrypted payment processing and never store your full card number "
                "on our servers.",
            ),
        ],
    ),
    (
        "Support",
        [
            (
                "How do I contact customer support?",
                "You can reach us by live chat on our website, by email at support at acmewidgets "
                "dot com, or by phone at one eight hundred five five five zero one two three. We "
                "are available Monday to Friday, nine to five.",
            ),
            (
                "What are your support hours?",
                "Support is available Monday through Friday, from nine in the morning to five in "
                "the evening, Eastern Time.",
            ),
        ],
    ),
]


def _text(pdf: FPDF, text: str, size: int, style: str = "", color=(0, 0, 0)) -> None:
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, size * 0.6, text, new_x="LMARGIN", new_y="NEXT")


def build() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    pdf.add_page()
    _text(pdf, "Acme Widgets - Customer FAQ", 20, "B")
    _text(pdf, "Answers to common customer questions.", 10, color=(120, 120, 120))

    for section, faqs in SECTIONS:
        pdf.ln(4)
        _text(pdf, section, 14, "B")
        for question, answer in faqs:
            _text(pdf, question, 11, "B")
            _text(pdf, answer, 11)
            pdf.ln(2)

    OUT.write_bytes(pdf.output())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()