import pdfplumber
import io

def process_pdf(path, start_page, top_margin, bottom_margin, left_margin, right_margin):
    all_text = io.StringIO()

    with pdfplumber.open(path) as pdf:
        start_page -= 1  # Adjust for zero-indexing

        for page in pdf.pages[start_page:]:
            bbox = (
                left_margin,
                top_margin,
                page.width - right_margin,
                page.height - bottom_margin
            )
            cropped_page = page.within_bbox(bbox)
            text = cropped_page.extract_text()

            if text:
                all_text.write(text + "\n")

    return all_text.getvalue()