import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, keeping each page separate.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list containing the page number and extracted text
        for each page.
    """

    pdf_file = Path(pdf_path)

    # Check whether the PDF exists.
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    extracted_pages = []

    # Open the PDF.
    with pymupdf.open(pdf_file) as document:

        # Process each page individually.
        for page_number, page in enumerate(document, start=1):

            # Extract the readable text from this page.
            text = page.get_text("text")

            # Store the page number and its text.
            extracted_pages.append({
                "page": page_number,
                "text": text
            })

    return extracted_pages


if __name__ == "__main__":

    pdf_path = "data/papers/NeurIPS-2024-contrasting-with-symile-simple-model-agnostic-representation-learning-for-unlimited-modalities-Paper-Conference.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"Total pages: {len(pages)}")

    # Display the text from the first page.
    if pages:
        print("\n--- First Page ---\n")
        print(pages[0]["text"])