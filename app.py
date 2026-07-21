import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import io
from datetime import datetime
import math

# Page setup
st.set_page_config(page_title="Receptacle Counter", page_icon="🔌", layout="wide")

st.title("🔌 Electrical Receptacle Counter")
st.write("Detects and counts electrical receptacle symbols in architectural drawings")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Detection Settings")
    
    match_threshold = st.slider(
        "Match Sensitivity", 
        0.3, 0.95, 0.5, 0.05,
        help="Lower = find more (may include false positives). Higher = stricter matching"
    )
    
    min_symbol_size = st.slider(
        "Minimum Symbol Size (pixels²)",
        30, 500, 80, 10,
        help="Smallest area to consider as a symbol"
    )
    
    max_symbol_size = st.slider(
        "Maximum Symbol Size (pixels²)",
        500, 8000, 4000, 100,
        help="Largest area to consider as a symbol"
    )
    
    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. Finds symbols near text in legend
    2. Filters out letters and text
    3. Identifies receptacle features (wires, connections)
    4. Searches power plan for matches
    5. Unifies overlapping detections
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Receptacles often have lines/wires passing through
    - Letters are automatically filtered out
    - Lower sensitivity if missing symbols
    - Higher sensitivity if getting false matches
    """)

# Upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Step 1: Upload Legend Page")
    st.caption("Page showing receptacle symbol definitions")
    legend_file = st.file_uploader("Legend image", type=['png','jpg','jpeg','tiff','bmp'])
    if legend_file:
        st.image(legend_file, use_container_width=True)

with col2:
    st.subheader("🔌 Step 2: Upload Power Plan")
    st.caption("Page where receptacles will be counted")
    power_file = st.file_uploader("Power plan image", type=['png','jpg','jpeg','tiff','bmp'])
    if power_file:
        st.image(power_file, use_container_width=True)

# Processing
if legend_file and power_file:
    st.markdown("---")
    
    if st.button("🔍 Count Receptacles", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        progress_text = st.empty()
        detail_text = st.empty()
        step_status = st.empty()
        
        # ===== STEP 1: Load & Preprocess (0-15%) =====
        progress_text.markdown("**📥 Step 1/8: Loading and preprocessing images...**")
        
        for p in range(0, 15, 3):
            progress_bar.progress(p)
            detail_text.text(f"Loading... {p}%")
        
        legend_gray = Image.open(legend_file).convert('L')
        power_gray = Image.open(power_file).convert('L')
        power_color = Image.open(power_file).convert('RGB')
        
        progress_bar.progress(15)
        detail_text.text("✅ Images loaded")
        step_status.success("Step 1 complete: Images loaded")
        
        # ===== STEP 2: Enhance & Binarize (15-25%) =====
        progress_text.markdown("**🔧 Step 2/8: Enhancing image quality...**")
        
        progress_bar.progress(17)
        detail_text.text("Sharpening...")
        legend_gray = legend_gray.filter(ImageFilter.SHARPEN)
        power_gray = power_gray.filter(ImageFilter.SHARPEN)
        
        progress_bar.progress(20)
        detail_text.text("Converting to binary...")
        
        # Adaptive-like thresholding
        legend_arr = np.array(legend_gray)
        power_arr = np.array(power_gray)
        
        # Use local mean for better thresholding
        def adaptive_threshold(img_arr, window_size=15):
            from scipy import ndimage
            # Use simple mean filter
            mean = ndimage.uniform_filter(img_arr.astype(float), size=window_size)
            binary = (img_arr < mean - 5).astype(np.uint8)
            return binary
        
        try:
            legend_bin = adaptive_threshold(legend_arr)
            power_bin = adaptive_threshold(power_arr)
        except:
            # Fallback to simple threshold
            legend_bin = (legend_arr < 128).astype(np.uint8)
            power_bin = (power_arr < 128).astype(np.uint8)
        
        h_legend, w_legend = legend_bin.shape
        h_power, w_power = power_bin.shape
        
        progress_bar.progress(25)
        detail_text.text(f"Legend: {w_legend}×{h_legend} | Power Plan: {w_power}×{h_power}")
        step_status.success("Step 2 complete: Images enhanced")
        
        # ===== STEP 3: Find Text Regions (25-35%) =====
        progress_text.markdown("**📝 Step 3/8: Identifying text regions in legend...**")
        
        progress_bar.progress(27)
        detail_text.text("Analyzing row density...")
        
        # Find rows with text (consistent black pixel patterns)
        row_black_ratio = np.sum(legend_bin == 1, axis=1) / w_legend
        
        # Text rows have moderate black pixel density (5-50%)
        text_mask = (row_black_ratio > 0.03) & (row_black_ratio < 0.5)
        text_rows = np.where(text_mask)[0]
        
        progress_bar.progress(30)
        detail_text.text(f"Found {len(text_rows)} text rows")
        
        # Group into bands
        text_bands = []
        if len(text_rows) > 0:
            current = [text_rows[0]]
            for r in text_rows[1:]:
                if r - current[-1] <= 4:
                    current.append(r)
                else:
                    if len(current) > 10:
                        text_bands.append((min(current), max(current)))
                    current = [r]
            if len(current) > 10:
                text_bands.append((min(current), max(current)))
        
        progress_bar.progress(33)
        detail_text.text(f"Found {len(text_bands)} text bands")
        
        # ===== STEP 4: Find Letters to Filter Out (33-40%) =====
        progress_text.markdown("**🔤 Step 4/8: Identifying letters to exclude...**")
        
        progress_bar.progress(35)
        detail_text.text("Finding text characters...")
        
        # Find all small components that are likely letters
        letter_positions = []
        
        for band_y1, band_y2 in text_bands:
            band = legend_bin[band_y1:band_y2+1, :]
            
            # Look at middle-right portion (where text usually is)
            text_region = band[:, int(w_legend*0.4):]
            
            # Find connected components
            visited = np.zeros_like(text_region, dtype=bool)
            
            for y in range(text_region.shape[0]):
                for x in range(text_region.shape[1]):
                    if text_region[y,x] == 1 and not visited[y,x]:
                        # Flood fill
                        stack = [(y,x)]
                        pixels = []
                        min_x, min_y = x, y
                        max_x, max_y = x, y
                        
                        while stack:
                            cy, cx = stack.pop()
                            if (0 <= cy < text_region.shape[0] and 
                                0 <= cx < text_region.shape[1] and 
                                text_region[cy,cx] == 1 and 
                                not visited[cy,cx]):
                                
                                visited[cy,cx] = True
                                pixels.append((cy,cx))
                                min_x = min(min_x, cx)
                                min_y = min(min_y, cy)
                                max_x = max(max_x, cx)
                                max_y = max(max_y, cy)
                                
                                for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1),
                                              (cy-1,cx-1),(cy-1,cx+1),(cy+1,cx-1),(cy+1,cx+1)]:
                                    stack.append((ny,nx))
                        
                        w_char = max_x - min_x + 1
                        h_char = max_y - min_y + 1
                        aspect = w_char / h_char if h_char > 0 else 0
                        
                        # Letters are typically small and have certain aspect ratios
                        if (10 < w_char < 80 and 10 < h_char < 80 and 
                            0.2 < aspect < 2.0 and len(pixels) < 2000):
                            
                            # Calculate character features
                            density = len(pixels) / (w_char * h_char)
                            
                            # Letters typically have 20-60% fill ratio
                            if 0.15 < density < 0.65:
                                letter_positions.append({
                                    'x': int(w_legend*0.4) + min_x,
                                    'y': band_y1 + min_y,
                                    'w': w_char,
                                    'h': h_char,
                                    'area': len(pixels),
                                    'density': density
                                })
        
        progress_bar.progress(40)
        detail_text.text(f"Identified {len(letter_positions)} letter characters to exclude")
        step_status.success("Step 4 complete: Letters identified")
        
        # ===== STEP 5: Extract Symbols (40-55%) =====
        progress_text.markdown("**🔌 Step 5/8: Extracting receptacle symbols from legend...**")
        
        templates = []
        
        for band_idx, (band_y1, band_y2) in enumerate(text_bands):
            progress = 40 + int((band_idx / max(1, len(text_bands))) * 10)
            progress_bar.progress(progress)
            detail_text.text(f"Processing band {band_idx+1}/{len(text_bands)}...")
            
            band = legend_bin[band_y1:band_y2+1, :]
            
            # Look at LEFT side for symbols
            left_width = int(w_legend * 0.35)
            left_band = band[:, :left_width]
            
            # Find components
            visited = np.zeros_like(left_band, dtype=bool)
            
            for y in range(left_band.shape[0]):
                for x in range(left_band.shape[1]):
                    if left_band[y,x] == 1 and not visited[y,x]:
                        # Flood fill
                        stack = [(y,x)]
                        pixels = []
                        min_x, min_y = x, y
                        max_x, max_y = x, y
                        
                        while stack:
                            cy, cx = stack.pop()
                            if (0 <= cy < left_band.shape[0] and 
                                0 <= cx < left_band.shape[1] and 
                                left_band[cy,cx] == 1 and 
                                not visited[cy,cx]):
                                
                                visited[cy,cx] = True
                                pixels.append((cy,cx))
                                min_x = min(min_x, cx)
                                min_y = min(min_y, cy)
                                max_x = max(max_x, cx)
                                max_y = max(max_y, cy)
                                
                                for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1),
                                              (cy-1,cx-1),(cy-1,cx+1),(cy+1,cx-1),(cy+1,cx+1)]:
                                    stack.append((ny,nx))
                        
                        comp_w = max_x - min_x + 1
                        comp_h = max_y - min_y + 1
                        comp_area = len(pixels)
                        
                        # Check if it could be a symbol (not a letter)
                        if min_symbol_size < comp_area < max_symbol_size:
                            aspect = comp_w / comp_h if comp_h > 0 else 0
                            
                            # Receptacles tend to be more square than letters
                            if 0.4 < aspect < 2.5:
                                
                                # Check if this component overlaps with letter positions
                                comp_x = min_x
                                comp_y = band_y1 + min_y
                                
                                is_letter = False
                                for letter in letter_positions:
                                    # Check overlap
                                    if (abs(comp_x - letter['x']) < 20 and 
                                        abs(comp_y - letter['y']) < 20):
                                        is_letter = True
                                        break
                                
                                if not is_letter:
                                    # Extract with padding
                                    pad = 4
                                    ty1 = max(0, min_y - pad)
                                    ty2 = min(left_band.shape[0], max_y + pad + 1)
                                    tx1 = max(0, min_x - pad)
                                    tx2 = min(left_band.shape[1], max_x + pad + 1)
                                    
                                    template_img = left_band[ty1:ty2, tx1:tx2]
                                    
                                    if template_img.shape[0] > 8 and template_img.shape[1] > 8:
                                        # Calculate features to identify receptacles
                                        th, tw = template_img.shape
                                        
                                        # Check for horizontal lines (wires/connections)
                                        h_lines = 0
                                        for row in range(th):
                                            row_sum = np.sum(template_img[row, :])
                                            if row_sum > tw * 0.3:  # Line spanning >30% of width
                                                h_lines += 1
                                        
                                        # Check for vertical lines
                                        v_lines = 0
                                        for col in range(tw):
                                            col_sum = np.sum(template_img[:, col])
                                            if col_sum > th * 0.3:
                                                v_lines += 1
                                        
                                        # Check for circular features
                                        center_y, center_x = th//2, tw//2
                                        circle_score = 0
                                        if th > 4 and tw > 4:
                                            for angle in range(0, 360, 30):
                                                rad = math.radians(angle)
                                                r = min(th, tw) // 3
                                                cy = int(center_y + r * math.sin(rad))
                                                cx = int(center_x + r * math.cos(rad))
                                                if 0 <= cy < th and 0 <= cx < tw:
                                                    if template_img[cy, cx] == 1:
                                                        circle_score += 1
                                        
                                        templates.append({
                                            'image': template_img,
                                            'w': comp_w,
                                            'h': comp_h,
                                            'area': comp_area,
                                            'aspect': aspect,
                                            'h_lines': h_lines,
                                            'v_lines': v_lines,
                                            'circle_score': circle_score,
                                            'band': band_idx,
                                            'position': (comp_x, comp_y)
                                        })
        
        progress_bar.progress(55)
        detail_text.text(f"✅ Extracted {len(templates)} potential symbols (letters excluded)")
        step_status.success(f"Step 5 complete: {len(templates)} symbols found")
        
        # Show extracted templates
        if templates:
            st.markdown("#### 📋 Extracted Symbols from Legend:")
            cols = st.columns(min(6, len(templates)))
            for i, t in enumerate(templates[:12]):
                with cols[i % 6]:
                    timg = Image.fromarray(t['image'].astype(np.uint8) * 255)
                    st.image(timg, caption=f"#{i+1} ({t['w']}×{t['h']})", use_container_width=True)
        
        # ===== STEP 6: Prioritize Receptacle-Like Templates (55-60%) =====
        progress_text.markdown("**🎯 Step 6/8: Identifying best receptacle candidates...**")
        
        progress_bar.progress(57)
        detail_text.text("Scoring templates by receptacle features...")
        
        # Score each template for "receptacle-ness"
        for t in templates:
            # Receptacles often have:
            # - Horizontal lines (wires passing through)
            # - Vertical lines (connection points)
            # - Circular elements (outlet shape)
            # - Moderate aspect ratio
            
            score = 0
            
            # Horizontal lines are very common in receptacles
            if t['h_lines'] >= 1:
                score += 3
            if t['h_lines'] >= 2:
                score += 2
            
            # Vertical lines suggest connections
            if t['v_lines'] >= 1:
                score += 2
            
            # Circle features suggest outlet shape
            if t['circle_score'] >= 6:
                score += 2
            if t['circle_score'] >= 9:
                score += 1
            
            # Good aspect ratio
            if 0.6 < t['aspect'] < 1.7:
                score += 1
            
            # Penalize if too letter-like (high density)
            density = t['area'] / (t['w'] * t['h']) if t['w'] * t['h'] > 0 else 0
            if density > 0.7:  # Very dense = probably letter
                score -= 2
            if density < 0.15:  # Very sparse = probably not symbol
                score -= 1
            
            t['receptacle_score'] = score
        
        # Sort by receptacle score
        templates.sort(key=lambda t: t['receptacle_score'], reverse=True)
        
        # Filter out low-scoring templates
        good_templates = [t for t in templates if t['receptacle_score'] >= 2]
        
        if not good_templates:
            good_templates = templates[:5]  # Fallback to top 5
        
        progress_bar.progress(60)
        detail_text.text(f"Selected {len(good_templates)} best receptacle candidates")
        step_status.success(f"Step 6 complete: {len(good_templates)} templates selected")
        
        # ===== STEP 7: Search Power Plan (60-85%) =====
        progress_text.markdown("**🔍 Step 7/8: Searching power plan for receptacles...**")
        
        step_size = max(2, min(h_power, w_power) // 100)
        all_detections = []
        
        for tidx, template in enumerate(good_templates[:8]):
            progress = 60 + int((tidx / max(1, len(good_templates[:8]))) * 20)
            progress_bar.progress(progress)
            detail_text.text(f"Searching template {tidx+1}/{min(8, len(good_templates))}... Found {len(all_detections)} matches")
            
            timg = template['image']
            th, tw = timg.shape
            
            scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
            
            for scale in scales:
                new_h = int(th * scale)
                new_w = int(tw * scale)
                
                if new_h < 6 or new_w < 6:
                    continue
                if new_h > h_power or new_w > w_power:
                    continue
                
                # Resize template
                tpil = Image.fromarray(timg.astype(np.uint8) * 255)
                tscaled = np.array(tpil.resize((new_w, new_h), Image.Resampling.LANCZOS)) > 128
                tscaled = tscaled.astype(np.uint8)
                
                # Count black pixels in template
                template_black = np.sum(tscaled == 1)
                if template_black < 5:
                    continue
                
                # Sliding window
                for y in range(0, h_power - new_h + 1, step_size):
                    for x in range(0, w_power - new_w + 1, step_size):
                        patch = power_bin[y:y+new_h, x:x+new_w]
                        
                        if patch.shape != tscaled.shape:
                            continue
                        
                        # Match black pixels only
                        black_mask = (tscaled == 1)
                        patch_black = np.sum(patch[black_mask] == 1)
                        
                        match_ratio = patch_black / template_black
                        
                        if match_ratio >= match_threshold:
                            all_detections.append({
                                'x': int(x),
                                'y': int(y),
                                'w': new_w,
                                'h': new_h,
                                'conf': float(match_ratio),
                                'template_id': tidx,
                                'template_score': template['receptacle_score']
                            })
        
        progress_bar.progress(82)
        detail_text.text(f"Raw matches found: {len(all_detections)}")
        
        # ===== STEP 8: Unify & Remove Duplicates (82-100%) =====
        progress_text.markdown("**🎯 Step 8/8: Unifying detections and removing duplicates...**")
        
        progress_bar.progress(85)
        detail_text.text("Clustering nearby detections...")
        
        if all_detections:
            # Sort by confidence * template_score
            all_detections.sort(key=lambda d: d['conf'] * (1 + d['template_score'] * 0.1), reverse=True)
            
            # Cluster overlapping detections
            clusters = []
            used = set()
            
            for i, d in enumerate(all_detections):
                if i in used:
                    continue
                
                cluster = [d]
                used.add(i)
                
                # Find all detections overlapping with this one
                for j, d2 in enumerate(all_detections):
                    if j in used:
                        continue
                    
                    # Check overlap
                    ox1 = max(d['x'], d2['x'])
                    oy1 = max(d['y'], d2['y'])
                    ox2 = min(d['x']+d['w'], d2['x']+d2['w'])
                    oy2 = min(d['y']+d['h'], d2['y']+d2['h'])
                    
                    if ox2 > ox1 and oy2 > oy1:
                        overlap_area = (ox2-ox1) * (oy2-oy1)
                        d_area = min(d['w']*d['h'], d2['w']*d2['h'])
                        
                        if overlap_area / d_area > 0.3:
                            cluster.append(d2)
                            used.add(j)
                
                clusters.append(cluster)
            
            progress_bar.progress(90)
            detail_text.text(f"Found {len(clusters)} unique detection clusters")
            
            # For each cluster, keep the best detection
            unified = []
            
            for cluster in clusters:
                # Choose the one with highest confidence * score
                best = max(cluster, key=lambda d: d['conf'] * (1 + d['template_score'] * 0.1))
                
                # Average position of all detections in cluster (weighted by confidence)
                total_conf = sum(d['conf'] for d in cluster)
                if total_conf > 0:
                    avg_x = int(sum(d['x'] * d['conf'] for d in cluster) / total_conf)
                    avg_y = int(sum(d['y'] * d['conf'] for d in cluster) / total_conf)
                    avg_w = int(sum(d['w'] * d['conf'] for d in cluster) / total_conf)
                    avg_h = int(sum(d['h'] * d['conf'] for d in cluster) / total_conf)
                else:
                    avg_x, avg_y = best['x'], best['y']
                    avg_w, avg_h = best['w'], best['h']
                
                unified.append({
                    'x': avg_x,
                    'y': avg_y,
                    'w': avg_w,
                    'h': avg_h,
                    'conf': best['conf'],
                    'cluster_size': len(cluster),
                    'best_score': best['template_score']
                })
            
            detections = unified
        else:
            detections = []
        
        progress_bar.progress(95)
        detail_text.text(f"Unified to {len(detections)} unique receptacles")
        
        # Draw results with UNIFIED markers
        progress_bar.progress(97)
        detail_text.text("Drawing unified markers...")
        
        draw = ImageDraw.Draw(power_color)
        
        # Try to load font
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        for i, d in enumerate(detections):
            x1, y1 = d['x'], d['y']
            x2, y2 = d['x'] + d['w'], d['y'] + d['h']
            
            # RED rectangle - thick border
            draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline='red', width=4)
            draw.rectangle([x1, y1, x2, y2], outline='#ff0000', width=2)
            
            # Red semi-transparent fill
            for dy in range(d['h']):
                for dx in range(0, d['w'], 3):
                    if (dx + dy) % 6 < 3:
                        try:
                            pixel = power_color.getpixel((x1+dx, y1+dy))
                            power_color.putpixel((x1+dx, y1+dy), 
                                               (min(255, pixel[0]+40), pixel[1], pixel[2]))
                        except:
                            pass
            
            # Number label with white background
            label = f"R{i+1}"
            text_bbox = draw.textbbox((x1+3, y1-22), label, font=font_large)
            # Draw white background
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, 
                          text_bbox[2]+2, text_bbox[3]+2], fill='white')
            # Draw red text
            draw.text((x1+3, y1-22), label, fill='red', font=font_large)
            
            # Confidence below
            conf_label = f"{d['conf']:.0%}"
            draw.text((x1+3, y1+2), conf_label, fill='red', font=font_small)
        
        progress_bar.progress(100)
        detail_text.text("✅ Detection complete!")
        step_status.success("All steps complete!")
        
        # ===== DISPLAY RESULTS =====
        st.markdown("---")
        
        if detections:
            # Big red result box
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%); 
                        padding: 30px; border-radius: 15px; 
                        border: 4px solid #ff0000; text-align: center; margin: 20px 0;
                        box-shadow: 0 4px 15px rgba(255,0,0,0.2);">
                <h2 style="color: #cc0000; margin: 0;">🔌 RECEPTACLES DETECTED</h2>
                <h1 style="color: #ff0000; font-size: 90px; margin: 15px 0; font-weight: 900;">
                    {len(detections)}
                </h1>
                <h3 style="color: #cc0000; margin: 0;">Unique Receptacle Symbols Found</h3>
                <p style="color: #666; margin-top: 10px;">
                    Each marked with red rectangle and "R" number
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show result image
            st.image(power_color, 
                    caption=f"🔴 {len(detections)} Receptacles Marked in RED (no duplicates)",
                    use_container_width=True)
            
            # Download options
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                buf = io.BytesIO()
                power_color.save(buf, format='PNG', quality=95)
                st.download_button(
                    label="📥 Download Marked Image",
                    data=buf.getvalue(),
                    file_name=f"receptacles_marked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_b:
                # Stats
                st.markdown("### 📊 Quality Stats")
                confs = [d['conf'] for d in detections]
                st.metric("Avg Confidence", f"{np.mean(confs):.1%}")
                st.metric("Max Confidence", f"{np.max(confs):.1%}")
                st.metric("Detections", len(detections))
            
            with col_c:
                st.markdown("### 🎯 Detection Quality")
                high = sum(1 for d in detections if d['conf'] > 0.8)
                med = sum(1 for d in detections if 0.5 <= d['conf'] <= 0.8)
                low = sum(1 for d in detections if d['conf'] < 0.5)
                st.write(f"🟢 High confidence: {high}")
                st.write(f"🟡 Medium confidence: {med}")
                st.write(f"🔴 Low confidence: {low}")
            
            # Detailed table
            st.markdown("### 📋 Unified Detection List (No Duplicates)")
            
            # Create table
            table_data = []
            for i, d in enumerate(detections):
                table_data.append({
                    "Mark": f"R{i+1}",
                    "Position": f"({d['x']}, {d['y']})",
                    "Size": f"{d['w']}×{d['h']}",
                    "Confidence": f"{d['conf']:.1%}",
                    "Template Score": f"{d['best_score']}/8",
                    "Cluster Size": d['cluster_size']
                })
            
            st.dataframe(table_data, use_container_width=True, hide_index=True)
            
            st.info(f"""
            ✅ **Unification Summary:**
            - {len(detections)} unique receptacles identified
            - Each marked with RED rectangle and "R" number
            - No duplicate markings (nearby detections merged)
            - Letters and text filtered out
            - Only receptacle-like symbols counted
            """)
        
        else:
            st.warning("""
            ### ⚠️ No Receptacles Found
            
            **Suggestions:**
            1. **Lower the sensitivity** (try 0.3 or 0.4)
            2. **Check legend image** - symbols should be clear
            3. **Use higher resolution** images
            4. **Symbols must match** between legend and power plan
            5. **Ensure good contrast** - dark symbols on light background
            """)

else:
    st.info("""
    ### 👆 Getting Started
    
    1. **Upload Legend Page** - The electrical legend showing receptacle symbols
    2. **Upload Power Plan** - The floor plan where receptacles should be counted
    3. **Click "Count Receptacles"** to start detection
    
    The app will:
    - Find symbols near text in the legend
    - Filter out letters and text
    - Search the power plan for matches
    - Mark all receptacles with unified RED markers
    """)
