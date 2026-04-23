from pathlib import Path
import re


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 54
RIGHT_MARGIN = 54
TOP_MARGIN = 54
BOTTOM_MARGIN = 54
BODY_FONT_SIZE = 11
HEADING_FONT_SIZE = 18
SUBHEADING_FONT_SIZE = 14
LINE_SPACING = 15
MAX_CHARS = 92


def escape_pdf_text(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_text(text, width):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def parse_markdown(md_text):
    elements = []
    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            elements.append(("blank", ""))
            continue
        if line.startswith("# "):
            elements.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            elements.append(("h2", line[3:].strip()))
        elif line.startswith("- "):
            elements.append(("bullet", line[2:].strip()))
        else:
            elements.append(("p", re.sub(r"\s+", " ", line.strip())))
    return elements


def emit_text_line(stream, x, y, font_name, font_size, text):
    safe_text = escape_pdf_text(text)
    stream.append(f"BT /{font_name} {font_size} Tf 1 0 0 1 {x} {y} Tm ({safe_text}) Tj ET")


def build_pdf_objects(markdown_text):
    elements = parse_markdown(markdown_text)
    pages = []
    stream = []
    y = PAGE_HEIGHT - TOP_MARGIN

    def flush_page():
        nonlocal stream, y
        if stream:
            pages.append("\n".join(stream) + "\n")
        stream = []
        y = PAGE_HEIGHT - TOP_MARGIN

    def ensure_space(lines_needed=1):
        nonlocal y
        required = lines_needed * LINE_SPACING
        if y - required < BOTTOM_MARGIN:
            flush_page()

    for kind, text in elements:
        if kind == "blank":
            y -= LINE_SPACING * 0.6
            continue

        if kind == "h1":
            ensure_space(2)
            emit_text_line(stream, LEFT_MARGIN, y, "F2", HEADING_FONT_SIZE, text)
            y -= LINE_SPACING * 1.8
            continue

        if kind == "h2":
            ensure_space(2)
            emit_text_line(stream, LEFT_MARGIN, y, "F2", SUBHEADING_FONT_SIZE, text)
            y -= LINE_SPACING * 1.4
            continue

        if kind == "bullet":
            wrapped = wrap_text(text, MAX_CHARS - 4)
            ensure_space(len(wrapped))
            for idx, line in enumerate(wrapped):
                prefix = "- " if idx == 0 else "  "
                emit_text_line(stream, LEFT_MARGIN, y, "F1", BODY_FONT_SIZE, prefix + line)
                y -= LINE_SPACING
            continue

        wrapped = wrap_text(text, MAX_CHARS)
        ensure_space(len(wrapped))
        for line in wrapped:
            emit_text_line(stream, LEFT_MARGIN, y, "F1", BODY_FONT_SIZE, line)
            y -= LINE_SPACING

    flush_page()

    objects = []

    def add_object(content):
        objects.append(content)
        return len(objects)

    font1_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font2_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_ids = []
    content_ids = []

    for page_stream in pages:
        stream_bytes = page_stream.encode("latin-1", errors="replace")
        content_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n"
            + page_stream +
            "endstream"
        )
        content_ids.append(add_object(content_obj))

    pages_id_placeholder = None
    for content_id in content_ids:
        page_obj = (
            "<< /Type /Page "
            f"/Parent PAGES_PLACEHOLDER 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        page_ids.append(add_object(page_obj))

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
    pages_id_placeholder = f"{pages_id} 0 R"

    for idx, page_id in enumerate(page_ids):
        objects[page_id - 1] = objects[page_id - 1].replace("PAGES_PLACEHOLDER", pages_id_placeholder)

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")
    return objects, catalog_id


def write_pdf(output_path, markdown_text):
    objects, catalog_id = build_pdf_objects(markdown_text)

    pdf = ["%PDF-1.4\n"]
    offsets = [0]
    current_offset = len(pdf[0].encode("latin-1"))

    for index, obj in enumerate(objects, start=1):
        obj_text = f"{index} 0 obj\n{obj}\nendobj\n"
        pdf.append(obj_text)
        offsets.append(current_offset)
        current_offset += len(obj_text.encode("latin-1"))

    xref_offset = current_offset
    pdf.append(f"xref\n0 {len(objects) + 1}\n")
    pdf.append("0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.append(f"{offset:010d} 00000 n \n")
    pdf.append(
        "trailer\n"
        f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    output_path.write_bytes("".join(pdf).encode("latin-1", errors="replace"))


def main():
    docs_dir = Path(__file__).resolve().parent
    md_path = docs_dir / "PacManDRL_Project_Report.md"
    pdf_path = docs_dir / "PacManDRL_Project_Report.pdf"
    write_pdf(pdf_path, md_path.read_text(encoding="utf-8"))
    print(pdf_path)


if __name__ == "__main__":
    main()
