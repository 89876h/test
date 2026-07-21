import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import io
from datetime import datetime

# Page setup
st.set_page_config(page_title="Receptacle Counter", page_icon="🔌", layout="wide")

st.title("🔌 Electrical Receptacle Counter")
st.write("Upload legend and power plan images to count electrical receptacles")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    match_threshold = st.slider("Match Sensitivity", 0.5, 0.95, 0.65, 0.05,
                               help="Higher = stricter matching")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Finds symbols near text in legend
    2. Matches them in power plan
    3. Counts all receptacles
    """)

# Upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Step 1: Upload Legend Page")
    legend_file = st.file_uploader("Legend image", type=['png','jpg','jpeg'])
    if legend_file:
        st.image(legend_file, use_container_width=True)

with col2:
    st.subheader("🔌 Step 2: Upload Power Plan")
    power_file = st.file_uploader("Power plan image", type=['png','jpg','jpeg'])
    if power_file:
        st.image(power_file, use_container_width=True)

# Processing
if legend_file and power_file:
    st.markdown("---")
    
    if st.button("🔍 Count Receptacles", type="primary", use_container_width=True):
        
        # Create progress display
        progress_bar = st.progress(0)
        progress_text = st.empty()
        detail_text = st.empty()
        
        # ===== STEP 1: Load Images (0-10%) =====
        progress_text.markdown("**Step 1/7: Loading images...**")
        progress_bar.progress(5)
        detail_text.text("Reading legend page...")
        
        legend = Image.open(legend_file).convert('L')
        power = Image.open(power_file).convert('L')
        power_color = Image.open(power_file).convert('RGB')
        
        progress_bar.progress(10)
        detail_text.text("✅ Images loaded successfully")
        
        # ===== STEP 2: Enhance Images (10-20%) =====
        progress_text.markdown("**Step 2/7: Enhancing image quality...**")
        progress_bar.progress(12)
        detail_text.text("Sharpening images...")
        
        legend = legend.filter(ImageFilter.SHARPEN)
        power = power.filter(ImageFilter.SHARPEN)
        
        progress_bar.progress(18)
        detail_text.text("Converting to black and white...")
        
        legend_bin = legend.point(lambda x: 0 if x < 128 else 255, '1')
        power_bin = power.point(lambda x: 0 if x < 128 else 255, '1')
        
        progress_bar.progress(20)
        detail_text.text("✅ Enhancement complete")
        
        # ===== STEP 3: Find Text Regions in Legend (20-35%) =====
        progress_text.markdown("**Step 3/7: Finding text and symbols in legend...**")
        progress_bar.progress(22)
        detail_text.text("Analyzing legend layout...")
        
        legend_arr = np.array(legend_bin)
        h, w = legend_arr.shape
        
        # Find rows with lots of black pixels (text rows)
        row_density = np.sum(legend_arr == 0, axis=1) / w
        
        # Find text bands
        text_rows = np.where(row_density > 0.05)[0]  # Rows with >5% black pixels
        
        progress_bar.progress(28)
        detail_text.text(f"Found {len(text_rows)} text rows...")
        
        # Group text rows into bands
        text_bands = []
        if len(text_rows) > 0:
            current_band = [text_rows[0]]
            for r in text_rows[1:]:
                if r - current_band[-1] <= 5:
                    current_band.append(r)
                else:
                    if len(current_band) > 8:
                        text_bands.append((min(current_band), max(current_band)))
                    current_band = [r]
            if len(current_band) > 8:
                text_bands.append((min(current_band), max(current_band)))
        
        progress_bar.progress(32)
        detail_text.text(f"Found {len(text_bands)} text bands with symbols")
        
        # ===== STEP 4: Extract Symbols Near Text (35-50%) =====
        progress_text.markdown("**Step 4/7: Extracting receptacle symbols...**")
        
        templates = []
        template_images = []
        
        for band_idx, (y1, y2) in enumerate(text_bands):
            # Update progress for each band
            progress = 35 + int((band_idx / max(1, len(text_bands))) * 12)
            progress_bar.progress(progress)
            detail_text.text(f"Processing text band {band_idx+1}/{len(text_bands)}...")
            
            # Extend band slightly
            y1_ext = max(0, y1 - 5)
            y2_ext = min(h, y2 + 5)
            
            # Get band region
            band = legend_arr[y1_ext:y2_ext+1, :]
            
            # Look on LEFT side for symbols (symbols usually left of text)
            left_width = int(w * 0.35)
            left_band = band[:, :left_width]
            
            # Find connected components in left band
            visited = np.zeros_like(left_band, dtype=bool)
            
            for y in range(left_band.shape[0]):
                for x in range(left_band.shape[1]):
                    if left_band[y, x] == 0 and not visited[y, x]:
                        # Flood fill
                        stack = [(y, x)]
                        pixels = []
                        min_x, min_y = x, y
                        max_x, max_y = x, y
                        
                        while stack:
                            cy, cx = stack.pop()
                            if (0 <= cy < left_band.shape[0] and 
                                0 <= cx < left_band.shape[1] and 
                                left_band[cy, cx] == 0 and 
                                not visited[cy, cx]):
                                
                                visited[cy, cx] = True
                                pixels.append((cy, cx))
                                
                                min_x = min(min_x, cx)
                                min_y = min(min_y, cy)
                                max_x = max(max_x, cx)
                                max_y = max(max_y, cy)
                                
                                # 8-connected neighbors
                                for ny, nx in [(cy-1,cx-1),(cy-1,cx),(cy-1,cx+1),
                                              (cy,cx-1),(cy,cx+1),
                                              (cy+1,cx-1),(cy+1,cx),(cy+1,cx+1)]:
                                    stack.append((ny, nx))
                        
                        # Filter by size (receptacle symbols are typically 15-100 pixels)
                        comp_width = max_x - min_x + 1
                        comp_height = max_y - min_y + 1
                        area = len(pixels)
                        
                        # Receptacle symbols are roughly square-ish
                        aspect_ratio = comp_width / comp_height if comp_height > 0 else 0
                        
                        if (50 < area < 5000 and 
                            10 < comp_width < 150 and 
                            10 < comp_height < 150 and
                            0.3 < aspect_ratio < 3.0):
                            
                            # Extract template with padding
                            pad = 3
                            ty1 = max(0, min_y - pad)
                            ty2 = min(left_band.shape[0], max_y + pad)
                            tx1 = max(0, min_x - pad)
                            tx2 = min(left_band.shape[1], max_x + pad)
                            
                            template = left_band[ty1:ty2+1, tx1:tx2+1]
                            
                            # Skip if too small
                            if template.shape[0] < 8 or template.shape[1] < 8:
                                continue
                            
                            templates.append({
                                'image': template,
                                'size': (comp_width, comp_height),
                                'area': area,
                                'band': band_idx
                            })
                            template_images.append(template)
        
        progress_bar.progress(50)
        detail_text.text(f"✅ Extracted {len(templates)} potential receptacle symbols")
        
        # Show extracted templates
        if templates:
            st.markdown("#### 📋 Extracted Symbols from Legend:")
            cols = st.columns(min(6, len(templates)))
            for i, t in enumerate(templates[:12]):
                with cols[i % 6]:
                    timg = Image.fromarray(t['image'].astype(np.uint8) * 255)
                    st.image(timg, caption=f"#{i+1}", use_container_width=True)
        
        # ===== STEP 5: Search in Power Plan (50-80%) =====
        progress_text.markdown("**Step 5/7: Searching power plan for receptacles...**")
        
        power_arr = np.array(power_bin)
        ph, pw = power_arr.shape
        detections = []
        
        # Use unique templates (remove duplicates)
        unique_templates = []
        used_sizes = []
        for t in templates:
            size_key = (t['size'][0]//5, t['size'][1]//5)
            if size_key not in used_sizes:
                unique_templates.append(t)
                used_sizes.append(size_key)
        
        templates = unique_templates[:8]  # Limit to 8 unique templates
        
        step_size = max(3, min(ph, pw) // 80)  # Adaptive step size
        
        for tidx, template in enumerate(templates):
            # Update progress
            progress = 50 + int((tidx / max(1, len(templates))) * 25)
            progress_bar.progress(progress)
            detail_text.text(f"Searching with symbol {tidx+1}/{len(templates)}... Found {len(detections)} matches")
            
            timg = template['image']
            th, tw = timg.shape
            
            # Multi-scale search
            scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
            
            for scale in scales:
                new_h = int(th * scale)
                new_w = int(tw * scale)
                
                if new_h < 8 or new_w < 8:
                    continue
                if new_h > ph or new_w > pw:
                    continue
                
                # Resize template
                tpil = Image.fromarray(timg.astype(np.uint8) * 255)
                tscaled = np.array(tpil.resize((new_w, new_h), Image.Resampling.LANCZOS)) // 255
                
                # Slide across power plan
                for y in range(0, ph - new_h, step_size):
                    for x in range(0, pw - new_w, step_size):
                        patch = power_arr[y:y+new_h, x:x+new_w]
                        
                        if patch.shape == tscaled.shape:
                            # Calculate match percentage
                            black_in_template = np.sum(tscaled == 0)
                            if black_in_template == 0:
                                continue
                            
                            # Only compare black pixels in template
                            black_mask = (tscaled == 0)
                            matching_black = np.sum(patch[black_mask] == 0)
                            match_ratio = matching_black / black_in_template
                            
                            if match_ratio >= match_threshold:
                                detections.append({
                                    'x': int(x),
                                    'y': int(y),
                                    'w': new_w,
                                    'h': new_h,
                                    'conf': float(match_ratio),
                                    'template_id': tidx
                                })
        
        progress_bar.progress(80)
        detail_text.text(f"✅ Search complete. Found {len(detections)} raw matches")
        
        # ===== STEP 6: Remove Duplicates (80-90%) =====
        progress_text.markdown("**Step 6/7: Removing duplicate detections...**")
        progress_bar.progress(82)
        detail_text.text(f"Filtering {len(detections)} candidates...")
        
        if detections:
            # Sort by confidence
            detections.sort(key=lambda d: d['conf'], reverse=True)
            
            # Non-maximum suppression
            final = []
            
            for i, d in enumerate(detections):
                if i % 100 == 0:
                    progress = 82 + int((i / len(detections)) * 6)
                    progress_bar.progress(progress)
                    detail_text.text(f"Filtering... {i}/{len(detections)}")
                
                # Check overlap with kept detections
                overlap = False
                for f in final:
                    # Calculate overlap percentage
                    ox1 = max(d['x'], f['x'])
                    oy1 = max(d['y'], f['y'])
                    ox2 = min(d['x']+d['w'], f['x']+f['w'])
                    oy2 = min(d['y']+d['h'], f['y']+f['h'])
                    
                    if ox2 > ox1 and oy2 > oy1:
                        overlap_area = (ox2-ox1) * (oy2-oy1)
                        d_area = d['w'] * d['h']
                        if overlap_area / d_area > 0.3:
                            overlap = True
                            break
                
                if not overlap:
                    final.append(d)
            
            detections = final
        
        progress_bar.progress(90)
        detail_text.text(f"✅ {len(detections)} unique receptacles identified")
        
        # ===== STEP 7: Draw Results (90-100%) =====
        progress_text.markdown("**Step 7/7: Creating result image...**")
        progress_bar.progress(92)
        detail_text.text("Drawing detection boxes...")
        
        # Draw on power plan
        draw = ImageDraw.Draw(power_color)
        
        # Try to load a font, or use default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for i, d in enumerate(detections):
            # RED rectangle
            x1, y1 = d['x'], d['y']
            x2, y2 = d['x']+d['w'], d['y']+d['h']
            
            # Draw red outline (3 pixels thick)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Draw red fill with transparency (for visibility)
            draw.rectangle([x1, y1, x2, y2], outline=None, fill=(255,0,0,30))
            
            # Draw number label in red
            label = f"#{i+1}"
            # Draw white background for text
            text_bbox = draw.textbbox((x1+2, y1-18), label, font=font)
            draw.rectangle(text_bbox, fill='white')
            draw.text((x1+2, y1-18), label, fill='red', font=font)
        
        progress_bar.progress(98)
        detail_text.text("✅ Drawing complete")
        
        # ===== SHOW RESULTS =====
        progress_bar.progress(100)
        progress_text.markdown("### ✅ Processing Complete!")
        detail_text.text("")
        
        st.markdown("---")
        
        if detections:
            # Big success message
            st.markdown(f"""
            <div style="background-color:#ffe6e6; padding:30px; border-radius:15px; 
                        border:3px solid red; text-align:center; margin:20px 0;">
                <h2 style="color:red;">🔌 Receptacles Detected</h2>
                <h1 style="color:red; font-size:80px; margin:10px 0;">{len(detections)}</h1>
                <h3 style="color:red;">Total Count</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show result image
            st.image(power_color, 
                    caption=f"🔴 {len(detections)} Receptacles Found (marked in RED)",
                    use_container_width=True)
            
            # Download buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Download image
                buf = io.BytesIO()
                power_color.save(buf, format='PNG')
                st.download_button(
                    label="📥 Download Marked Image (PNG)",
                    data=buf.getvalue(),
                    file_name=f"receptacles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_b:
                # Show stats
                st.markdown("### 📊 Detection Stats")
                confidences = [d['conf'] for d in detections]
                st.write(f"**Highest confidence:** {max(confidences):.1%}")
                st.write(f"**Average confidence:** {np.mean(confidences):.1%}")
                st.write(f"**Lowest confidence:** {min(confidences):.1%}")
            
            # Show table of first 20 detections
            st.markdown("### 📋 Detection Details (First 20)")
            
            table_html = """
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#ffcccc;">
                <th style="padding:8px; border:1px solid #ddd;">#</th>
                <th style="padding:8px; border:1px solid #ddd;">Position (X, Y)</th>
                <th style="padding:8px; border:1px solid #ddd;">Size (W×H)</th>
                <th style="padding:8px; border:1px solid #ddd;">Confidence</th>
            </tr>
            """
            
            for i, d in enumerate(detections[:20]):
                table_html += f"""
                <tr>
                    <td style="padding:8px; border:1px solid #ddd; text-align:center;">{i+1}</td>
                    <td style="padding:8px; border:1px solid #ddd; text-align:center;">({d['x']}, {d['y']})</td>
                    <td style="padding:8px; border:1px solid #ddd; text-align:center;">{d['w']}×{d['h']}</td>
                    <td style="padding:8px; border:1px solid #ddd; text-align:center;">{d['conf']:.1%}</td>
                </tr>
                """
            
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
            
            if len(detections) > 20:
                st.info(f"Showing 20 of {len(detections)} detections")
        
        else:
            # No detections
            st.warning("""
            ### ⚠️ No Receptacles Found
            
            **Try these fixes:**
            1. Lower the Match Sensitivity slider (try 0.5)
            2. Make sure legend has clear receptacle symbols
            3. Use higher resolution images (300 DPI)
            4. Ensure legend and power plan are from same project
            5. Symbols should be dark on light background
            """)

else:
    st.info("👆 Please upload both legend and power plan images to start")
