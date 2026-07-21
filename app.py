import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import cv2

st.set_page_config(page_title="Receptacle Counter", page_icon="🔌", layout="wide")
st.title("🔌 Electrical Receptacle Counter")
st.markdown("Structure-aware detection: Optimized for nested/complex symbols.")

# Initialize session state
if 'legend_confirmed' not in st.session_state:
    st.session_state.legend_confirmed = False
if 'confirmed_templates' not in st.session_state:
    st.session_state.confirmed_templates = []
if 'marked_legend' not in st.session_state:
    st.session_state.marked_legend = None
if 'all_symbols' not in st.session_state:
    st.session_state.all_symbols = []

with st.sidebar:
    st.header("⚙️ Settings")
    match_threshold = st.slider("Match Sensitivity", 0.3, 0.95, 0.6, 0.05)
    
    st.markdown("---")
    st.markdown("### 📋 Process")
    st.markdown("""
    **Step 1:** Upload legend  
    **Step 2:** Auto-extract symbols (left of text)  
    **Step 3:** Select receptacles  
    **Step 4:** Upload power plan  
    **Step 5:** Count  
    """)

# ===== STEP 1 & 2: UPLOAD & EXTRACT =====
st.header("📄 Step 1: Upload Legend Page")
legend_file = st.file_uploader("Upload legend image", type=['png', 'jpg', 'jpeg'], key='legend_upload')

if legend_file:
    legend_color = Image.open(legend_file).convert('RGB')
    legend_gray = np.array(legend_color.convert('L'))
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original Legend")
        st.image(legend_color, use_container_width=True)
    
    if st.button("🔍 Extract Symbols (Left of Text)", type="primary", use_container_width=True):
        with st.spinner("Analyzing legend structure..."):
            h, w = legend_gray.shape
            
            # 1. Find TEXT regions using horizontal projection
            binary = (legend_gray < 128).astype(np.uint8)
            row_sums = np.sum(binary, axis=1)
            
            # Identify rows that contain text
            text_threshold = np.mean(row_sums) * 0.3
            text_rows = np.where(row_sums > text_threshold)[0]
            
            # Group consecutive text rows into bands
            text_bands = []
            if len(text_rows) > 0:
                current_band = [text_rows[0]]
                for r in text_rows[1:]:
                    if r - current_band[-1] <= 5:
                        current_band.append(r)
                    else:
                        if len(current_band) > 3:
                            text_bands.append((min(current_band), max(current_band)))
                        current_band = [r]
                if len(current_band) > 3:
                    text_bands.append((min(current_band), max(current_band)))
            
            # 2. For EACH text band, find symbols to the LEFT
            raw_components = []
            
            for y1, y2 in text_bands:
                # Expand band significantly
                sy1 = max(0, y1 - 15)
                sy2 = min(h, y2 + 15)
                
                band_slice = binary[sy1:sy2, :]
                
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(band_slice, connectivity=8)
                
                for i in range(1, num_labels):
                    x, y, bw, bh, area = stats[i]
                    
                    # Skip text area (right side)
                    if x > w * 0.55:  
                        continue
                    
                    # Skip tiny noise
                    if area < 5 or bw < 3 or bh < 3:
                        continue
                    
                    # Store raw component info
                    raw_components.append({
                        'x': x, 'y': y, 'w': bw, 'h': bh, 
                        'area': area, 'band_y1': sy1, 'band_y2': sy2
                    })

            # ✅ FIX: MERGE NEARBY COMPONENTS (Handles Nested Rectangles)
            # If two components overlap or are very close, merge them into one bounding box
            merged_symbols = []
            used_indices = set()
            
            # Sort by X position to process left-to-right
            raw_components.sort(key=lambda c: c['x'])
            
            for i, comp in enumerate(raw_components):
                if i in used_indices:
                    continue
                
                # Start a new group with current component
                group = [comp]
                used_indices.add(i)
                
                # Check against all other unused components
                for j, other in enumerate(raw_components):
                    if j in used_indices:
                        continue
                    
                    # Calculate distance between components
                    # We check if they are roughly in same vertical area and close horizontally
                    cx1, cy1, cw1, ch1 = comp['x'], comp['y'], comp['w'], comp['h']
                    cx2, cy2, cw2, ch2 = other['x'], other['y'], other['w'], other['h']
                    
                    # Check for overlap or close proximity (gap < 10px)
                    x_dist = abs(cx1 - cx2)
                    y_dist = abs(cy1 - cy2)
                    
                    # If they are close enough to be part of same symbol
                    if x_dist < (max(cw1, cw2) + 10) and y_dist < (max(ch1, ch2) + 10):
                         # Also check if one is inside the other (nested rectangles)
                         # Or if they share significant vertical space
                         y_overlap = min(cy1+ch1, cy2+ch2) - max(cy1, cy2)
                         if y_overlap > 0 or y_dist < 5:
                             group.append(other)
                             used_indices.add(j)
                
                # Create a unified bounding box for the group
                min_x = min(c['x'] for c in group)
                min_y = min(c['y'] for c in group)
                max_x = max(c['x'] + c['w'] for c in group)
                max_y = max(c['y'] + c['h'] for c in group)
                
                final_w = max_x - min_x
                final_h = max_y - min_y
                
                # Add padding
                pad = 5
                cx1 = max(0, min_x - pad)
                cx2 = min(w, max_x + pad)
                cy1 = max(0, min_y - pad)
                cy2 = min(h, max_y + pad)
                
                # Crop from original gray image
                symbol_crop = legend_gray[cy1:cy2, cx1:cx2]
                
                # Normalize size
                target_h = 60
                scale = target_h / max(1, final_h)
                if final_h < 20: scale = max(scale, 3.0) # Upscale tiny symbols
                
                new_w = max(15, int(final_w * scale))
                new_h = int(final_h * scale)
                symbol_resized = cv2.resize(symbol_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                merged_symbols.append({
                    'image': symbol_resized,
                    'x': cx1, 'y': cy1, 'w': final_w, 'h': final_h,
                    'area': sum(c['area'] for c in group)
                })

            # Display Results
            symbols = merged_symbols
            marked_img = legend_color.copy()
            draw = ImageDraw.Draw(marked_img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            for idx, sym in enumerate(symbols):
                cx1, cy1 = sym['x'], sym['y']
                cx2, cy2 = cx1 + sym['w'], cy1 + sym['h']
                
                draw.rectangle([cx1-2, cy1-2, cx2+2, cy2+2], outline='lime', width=2)
                label = f"S{idx+1}"
                bbox = draw.textbbox((cx1, cy1-20), label, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='white')
                draw.text((cx1, cy1-20), label, fill='green', font=font)
            
            st.session_state.marked_legend = marked_img
            st.session_state.all_symbols = symbols
            st.session_state.text_bands = text_bands
            
            st.success(f"✅ Found {len(symbols)} symbols (Nested shapes merged)")
        
        with col2:
            st.subheader("Extracted Symbols (Green Boxes)")
            st.image(st.session_state.marked_legend, use_container_width=True)
            st.caption("Nested rectangles are now grouped as one symbol.")

# ===== STEP 3: SELECT RECEPTACLES =====
if st.session_state.get('all_symbols'):
    st.markdown("---")
    st.header("🔌 Step 3: Select Which Symbols Are Receptacles")
    
    symbols = st.session_state.all_symbols
    st.write(f"Found **{len(symbols)}** true symbols in legend")
    
    cols_per_row = 6
    rows = (len(symbols) + cols_per_row - 1) // cols_per_row
    
    selected_indices = []
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            sym_idx = row * cols_per_row + col_idx
            if sym_idx < len(symbols):
                sym = symbols[sym_idx]
                with cols[col_idx]:
                    # Robust display using BytesIO buffer
                    sym_gray = sym['image'].astype(np.uint8)
                    sym_pil = Image.fromarray(sym_gray, mode='L').convert('RGB')
                    
                    # Scale up for visibility
                    sym_pil = sym_pil.resize(
                        (max(sym_pil.width * 2, 50), max(sym_pil.height * 2, 50)), 
                        Image.NEAREST
                    )
                    
                    buf = io.BytesIO()
                    sym_pil.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.image(buf)
                    
                    st.caption(f"S{sym_idx+1}: {sym['w']}×{sym['h']}px")
                    
                    is_selected = st.checkbox(
                        "Receptacle",
                        key=f"sym_{sym_idx}",
                        value=False
                    )
                    if is_selected:
                        selected_indices.append(sym_idx)
    
    if st.button("✅ Confirm Selected Receptacles", type="primary", use_container_width=True):
        if selected_indices:
            st.session_state.confirmed_templates = [symbols[i] for i in selected_indices]
            st.session_state.legend_confirmed = True
            st.success(f"✅ **{len(selected_indices)} receptacles confirmed!**")
        else:
            st.warning("⚠️ Please select at least one symbol")

# ===== STEP 4 & 5: COUNT IN POWER PLAN =====
if st.session_state.legend_confirmed:
    st.markdown("---")
    st.header("🔌 Step 4: Upload Power Plan")
    power_file = st.file_uploader("Upload power plan", type=['png', 'jpg', 'jpeg'], key='power_upload')
    
    if power_file:
        power_color = Image.open(power_file).convert('RGB')
        power_gray = np.array(power_color.convert('L'))
        
        st.subheader("Power Plan")
        st.image(power_color, use_container_width=True)
        
        if st.button("🔍 Count Receptacles", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()
            
            templates = st.session_state.confirmed_templates
            power_bin = (power_gray < 128).astype(np.uint8)
            h_p, w_p = power_bin.shape
            
            all_detections = []
            
            for tidx, tmpl in enumerate(templates):
                progress = int((tidx / len(templates)) * 80)
                progress_bar.progress(progress)
                status.text(f"Matching template {tidx+1}/{len(templates)}...")
                
                timg = tmpl['image']
                th, tw = timg.shape
                
                for scale in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
                    new_h = int(th * scale)
                    new_w = int(tw * scale)
                    if new_h < 5 or new_w < 5 or new_h > h_p or new_w > w_p:
                        continue
                    
                    tpil = Image.fromarray(timg.astype(np.uint8))
                    tscaled = np.array(tpil.resize((new_w, new_h), Image.Resampling.LANCZOS))
                    tscaled_bin = (tscaled < 128).astype(np.uint8)
                    
                    black_count = np.sum(tscaled_bin)
                    if black_count < 3: continue
                    
                    step = max(2, min(new_h, new_w) // 3)
                    for y in range(0, h_p - new_h + 1, step):
                        for x in range(0, w_p - new_w + 1, step):
                            patch = power_bin[y:y+new_h, x:x+new_w]
                            mask = (tscaled_bin == 1)
                            match_black = np.sum(patch[mask])
                            ratio = match_black / black_count
                            
                            if ratio >= match_threshold:
                                all_detections.append({
                                    'x': x, 'y': y, 'w': new_w, 'h': new_h, 'conf': ratio
                                })
            
            progress_bar.progress(85)
            status.text("Removing duplicates...")
            
            if all_detections:
                all_detections.sort(key=lambda d: d['conf'], reverse=True)
                unified = []
                used = set()
                
                for i, d in enumerate(all_detections):
                    if i in used: continue
                    cluster = [d]
                    used.add(i)
                    
                    for j, d2 in enumerate(all_detections):
                        if j in used: continue
                        ox1 = max(d['x'], d2['x'])
                        oy1 = max(d['y'], d2['y'])
                        ox2 = min(d['x']+d['w'], d2['x']+d2['w'])
                        oy2 = min(d['y']+d['h'], d2['y']+d2['h'])
                        
                        if ox2 > ox1 and oy2 > oy1:
                            overlap = (ox2-ox1) * (oy2-oy1)
                            area = min(d['w']*d['h'], d2['w']*d2['h'])
                            if overlap / max(1, area) > 0.3:
                                cluster.append(d2)
                                used.add(j)
                    
                    total_conf = sum(c['conf'] for c in cluster)
                    avg_x = int(sum(c['x'] * c['conf'] for c in cluster) / max(1, total_conf))
                    avg_y = int(sum(c['y'] * c['conf'] for c in cluster) / max(1, total_conf))
                    avg_w = int(sum(c['w'] * c['conf'] for c in cluster) / max(1, total_conf))
                    avg_h = int(sum(c['h'] * c['conf'] for c in cluster) / max(1, total_conf))
                    
                    unified.append({'x': avg_x, 'y': avg_y, 'w': avg_w, 'h': avg_h, 'conf': max(c['conf'] for c in cluster)})
                
                detections = unified
            else:
                detections = []
            
            progress_bar.progress(95)
            status.text("Drawing results...")
            
            result_img = power_color.copy()
            draw = ImageDraw.Draw(result_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            for i, d in enumerate(detections):
                x1, y1, x2, y2 = d['x'], d['y'], d['x']+d['w'], d['y']+d['h']
                draw.rectangle([x1-2, y1-2, x2+2, y2+2], outline='red', width=3)
                label = f"R{i+1}"
                bbox = draw.textbbox((x1+3, y1-20), label, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='white')
                draw.text((x1+3, y1-20), label, fill='red', font=font)
            
            progress_bar.progress(100)
            status.text("✅ Complete!")
            
            st.markdown("---")
            if detections:
                st.markdown(f"""
                <div style="background:#fff5f5; padding:30px; border-radius:15px; 
                            border:4px solid red; text-align:center; margin:20px 0;">
                    <h2 style="color:#cc0000;">🔌 RECEPTACLES FOUND</h2>
                    <h1 style="color:red; font-size:80px; margin:15px 0;">{len(detections)}</h1>
                    <h3 style="color:#cc0000;">Total Count</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.image(result_img, caption=f"{len(detections)} Receptacles (RED markers)", use_container_width=True)
                
                buf = io.BytesIO()
                result_img.save(buf, format='PNG')
                st.download_button("📥 Download Result", buf.getvalue(), "receptacles_found.png", "image/png")
            else:
                st.warning("No receptacles found. Try lowering sensitivity.")
