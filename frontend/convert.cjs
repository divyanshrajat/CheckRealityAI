const HTMLtoJSX = require('htmltojsx');
const fs = require('fs');
const path = require('path');

const converter = new HTMLtoJSX({
  createClass: false,
});

const mappings = {
  'checkreality_ai_multi_modal_verification_platform': 'Dashboard',
  'analyze_upload_studio': 'Upload',
  'forensic_authenticity_report_xai': 'Report',
};

for (const [folder, comp_name] of Object.entries(mappings)) {
  const htmlPath = path.join(__dirname, '..', 'ui_ux', folder, 'code.html');
  if (fs.existsSync(htmlPath)) {
    let html = fs.readFileSync(htmlPath, 'utf8');
    
    // Extract everything inside <main>...</main>
    const mainMatch = html.match(/<main[^>]*>([\s\S]*?)<\/main>/i);
    if (mainMatch) {
      html = mainMatch[1];
    }
    
    // Remove script tags
    html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    
    // Use htmltojsx
    let jsxContent = converter.convert(html);
    
    jsxContent = jsxContent.replace(/onclick=/g, "onClick=");
    jsxContent = jsxContent.replace(/onsubmit=/g, "onSubmit=");
    jsxContent = jsxContent.replace(/oninput=/g, "onInput=");
    // Fix string/boolean issues like aria-hidden="true" which htmltojsx might make aria-hidden={true}
    jsxContent = jsxContent.replace(/=\{true\}/g, '="true"');
    jsxContent = jsxContent.replace(/=\{false\}/g, '="false"');
    
    const finalFile = `// @ts-nocheck
import React from 'react';
import { Link } from 'react-router-dom';

export default function ${comp_name}() {
  return (
    <div className="w-full flex flex-col">
      ${jsxContent}
    </div>
  );
}
`;
    
    fs.writeFileSync(path.join(__dirname, 'src', 'pages', `${comp_name}.tsx`), finalFile, 'utf8');
    console.log(`Converted ${comp_name}`);
  }
}
