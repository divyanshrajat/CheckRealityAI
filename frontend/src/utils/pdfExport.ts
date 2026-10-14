import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

/**
 * Generates and downloads a high-resolution, single-page A4 PDF certificate
 * directly to the user's machine.
 */
export async function downloadCertificatePdf(
  elementId: string = 'printable-forensic-report',
  filename: string = 'CheckRealityAI_Certificate.pdf'
): Promise<boolean> {
  const sourceElement = document.getElementById(elementId);
  if (!sourceElement) {
    console.error(`Certificate element #${elementId} not found in DOM`);
    return false;
  }

  // Create a clean off-screen clone to guarantee standard 800px width and full visibility
  const container = document.createElement('div');
  container.style.position = 'fixed';
  container.style.left = '-9999px';
  container.style.top = '0';
  container.style.width = '800px';
  container.style.backgroundColor = '#ffffff';
  container.style.zIndex = '-9999';
  container.style.overflow = 'visible';

  const clone = sourceElement.cloneNode(true) as HTMLElement;
  clone.style.display = 'block';
  clone.style.width = '800px';
  clone.style.maxWidth = '800px';
  clone.style.margin = '0 auto';
  clone.style.boxShadow = 'none';
  clone.style.backgroundColor = '#ffffff';

  container.appendChild(clone);
  document.body.appendChild(container);

  try {
    // Wait for fonts and assets to stabilize
    await new Promise((resolve) => setTimeout(resolve, 150));

    const canvas = await html2canvas(clone, {
      scale: 2, // 2x DPI for ultra-crisp typography and graphics
      useCORS: true,
      allowTaint: true,
      logging: false,
      backgroundColor: '#ffffff',
      width: 800,
      windowWidth: 1200,
    });

    const imgData = canvas.toDataURL('image/png', 1.0);

    // Create A4 PDF (210mm x 297mm)
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
      compress: true,
    });

    const pageWidth = 210;
    const pageHeight = 297;
    const marginX = 10;
    const contentWidth = pageWidth - marginX * 2; // 190mm
    const contentHeight = (canvas.height * contentWidth) / canvas.width;

    // Center vertically on A4 page if content height is within bounds
    const marginY = contentHeight < pageHeight ? Math.max(8, (pageHeight - contentHeight) / 5) : 8;

    pdf.addImage(imgData, 'PNG', marginX, marginY, contentWidth, Math.min(contentHeight, pageHeight - 16));

    // Direct browser file download
    pdf.save(filename);
    return true;
  } catch (error) {
    console.error('Error generating certificate PDF:', error);
    // Fallback to window.print() if canvas rendering is prevented
    window.print();
    return false;
  } finally {
    if (document.body.contains(container)) {
      document.body.removeChild(container);
    }
  }
}
