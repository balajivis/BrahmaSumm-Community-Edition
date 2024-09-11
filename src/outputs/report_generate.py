import os
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from PIL import Image as PILImage
from reportlab.lib import colors

def handle_li_tags(html_text, list_type="ul"):
    """
    Converts <li> HTML tags into ReportLab-compatible bullets or numbered lists.
    
    :param html_text: The HTML text containing <li> tags
    :param list_type: 'ul' for unordered list (bullets), 'ol' for ordered list (numbers)
    :return: Processed string with bullet points or numbering
    """
    if list_type == "ul":
        # Replace <li> tags with bullet points
        html_text = html_text.replace("<li>", '<br/>\u2022 ').replace("</li>", "")
    elif list_type == "ol":
        # Replace <li> tags with numbered list (this assumes items are numbered sequentially)
        items = html_text.split("<li>")
        for i, item in enumerate(items):
            if "</li>" in item:
                items[i] = f"<br/>{i}. " + item.split("</li>")[0]
        html_text = "".join(items)

    return html_text

def convert_html_to_reportlab_compatible(html_text):
    # Replace HTML tags with ReportLab-compatible tags
    html_text = html_text.replace("<h1>", '<br /><br /><b><font size="15">').replace("</h1>", "</font></b><br />")
    html_text = html_text.replace("<h2>", '<br /><br /><b><font size="13">').replace("</h2>", "</font></b><br />")
    html_text = html_text.replace("<p>", "<br />").replace("</p>", "<br />")  # Handle paragraphs
    html_text = handle_li_tags(html_text, list_type="ul")
    html_text = html_text.replace("<ul>", "").replace("</ul>", "")
    html_text = html_text.replace("<ol>", "").replace("</ol>", "")
    return html_text


def create_final_report(data: dict, report_path='reports/final_report.pdf'):
    """
    Generates a final report containing the analysis, summary, UMAP clusters, and themes.

    :param data: Dictionary containing necessary values for the report
    :param report_path: Path to save the final report PDF.
    """
    # 1. Retrieve data from the dictionary
    summary = convert_html_to_reportlab_compatible(data.get('summary', 'No summary available'))

    chunk_words = data.get('chunk_words', [])
    total_chunks = data.get('total_chunks', 0)
    total_words = data.get('total_words', 0)
    total_tokens = data.get('total_tokens', 0)
    tokens_sent_tokens = data.get('tokens_sent_tokens', 0)
    umap_image_path = data.get('umap_image_path', 'reports/umap_clusters.png')
    labels = data.get('labels', [])
    themes = data.get('themes', {})

    # Convert chunk words to a comma-separated string
    chunk_words_str = ", ".join(map(str, chunk_words))

    # 2. Create PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)

    # 3. Define styles
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    title_style = styles['Title']

    # 4. Build content for the PDF
    content = []

    # Title
    content.append(Paragraph("Document Analysis Report", title_style))
    content.append(Spacer(1, 0.25 * inch))

    # Summary of key numbers
    content.append(Paragraph(f"Total Chunks: {total_chunks}", normal_style))
    content.append(Paragraph(f"Total Words: {total_words}", normal_style))
    content.append(Paragraph(f"Total Tokens: {total_tokens}", normal_style))
    content.append(Paragraph(f"Tokens Sent to LLM: {tokens_sent_tokens}", normal_style))
    content.append(Spacer(1, 0.25 * inch))

    # Add the chunk words with wrapping
    # Commented for now...
    #chunk_words_para = Paragraph(f"Words per chunk: {chunk_words_str}", normal_style)
    #content.append(chunk_words_para)
    #content.append(Spacer(1, 0.25 * inch))

    # Add the summary with line breaks respected
    content.append(Paragraph("Summary:", title_style))
    summary_para = Paragraph(summary, normal_style)
    content.append(summary_para)
    content.append(PageBreak())

    # 5. Create the topic visualization
    clusters = {}
    for chunk_idx, cluster_label in enumerate(labels):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(f"{chunk_idx + 1}")

    # Sort the clusters by cluster label and also sort the chunk numbers within each cluster
    sorted_clusters = {cluster_label: sorted(chunks, key=lambda x: int(x)) for cluster_label, chunks in sorted(clusters.items())}

    print(sorted_clusters)
    # Create the data table for the PDF
    data_table = [["Theme", "Chunks"]]
    
    # Fill table with clusters, their themes, and chunk lists
    for cluster_label, chunks in clusters.items():
        theme = themes.get(cluster_label, f"Cluster {cluster_label}")
        theme_paragraph = Paragraph(theme, styles['Normal'])
        chunk_list = ", ".join(chunks)  # Concatenate all chunks into a single string
        chunks_paragraph = Paragraph(chunk_list, styles['Normal'])
        data_table.append([theme_paragraph, chunks_paragraph])
        
    #print(data_table)
    # Generate the PDF report
    doc = SimpleDocTemplate(report_path, pagesize=letter)

    # Add a title to the PDF
    title = Paragraph("Document Cluster Overview", styles['Title'])
    content.append(title)
    
    # Create the table
    table = Table(data_table, colWidths=[2 * inch, 4 * inch])
    
    # Add some style to the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    # Ensure text wrapping within each cell
    table.wrapOn(c, width, height)
    table.drawOn(c, 1 * inch, height - 6 * inch)  
    
    content.append(table)
    content.append(PageBreak())
    
    # 6. Insert UMAP cluster image (if it exists)
    if os.path.exists(umap_image_path):
        content.append(Paragraph("Topic themes and clusters", title_style))
        content.append(Spacer(1, 0.2 * inch))
        
        # GEt the aspect ratio of the image
        img = PILImage.open(umap_image_path)
        aspect_ratio = img.width / img.height
        img_width = width * 0.8
        img_height = img_width / aspect_ratio
        
        print(f"Image size: {img_width} x {img_height}")

        # Adjust image width and height to fit the page, keeping aspect ratio
        umap_image = Image(umap_image_path, img_width, img_height )
        content.append(umap_image)
        content.append(Spacer(1, 0.5 * inch))
        #content.append(PageBreak())

    # 6. Build the document
    doc.build(content)
    print(f"Report saved at {report_path}")