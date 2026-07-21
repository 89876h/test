import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import io
from datetime import datetime

# Page setup
st.set_page_config(page_title="Receptacle Counter", page_icon="🔌", layout="wide")

st.title("🔌 Electrical Receptacle Counter")
st.markdown("Detect and count electrical receptacles in architectural drawings")

# Initialize session state
if 'legend_confirmed' not in st.session_state:
    st.session_state.legend_confirmed = False
if 'confirmed_templates' not in st.session_state:
    st.session_state.confirmed_templates = []
if 'marked_legend' not in st.session_state:
    st.session_state.marked_legend = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    match_threshold = st.slider(
        "Match Sensitivity",
        0.3, 0.95, 0.5, 0.05,
        help="Lower = find more matches"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Process")
    st.markdown("""
    **Step 1:** Upload legend page
    **Step 2:** See symbols found in legend
    **Step 3:** Select which are receptacles
    **Step 4:** Upload power plan
    **Step 5:** Count receptacles
    """)

# ===== STEP 1: UPLOAD LEGEND =====
st.markdown("---")
st.header("📄 Step 1: Upload Legend Page")
st.caption("This page shows electrical symbols and their descriptions")

legend_file = st.file_uploader(
    "Upload legend image",
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    key='legend_upload'
)

if legend_file:
    # Load legend
    legend_color = Image.open(legend_file).convert('RGB')
    legend_gray = legend_color.convert('L')
    
    # Show original
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original Legend")
        st.image(legend_color, use_container_width=True)
    
    # ===== STEP 2: FIND ALL SYMBOLS IN LEGEND =====
    if st.button("🔍 Find Symbols in Legend", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing legend... Finding all symbols..."):
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Preprocess
            status.text("Enhancing image...")
            progress_bar.progress(10)
            
            legend_gray = legend_gray.filter(ImageFilter.SHARPEN)
            legend_arr = np.array(legend_gray)
            
            # Binary threshold
            status.text("Converting to binary...")
            progress_bar.progress(20)
            
            legend_bin = (legend_arr < 128).astype(np.uint8)
            
            h, w = legend_bin.shape
            
            # Find text rows
            status.text("Finding text regions...")
            progress_bar.progress(30)
            
            row_density = np.sum(legend_bin, axis=1) / w
            text_rows = np.where((row_density > 0.02) & (row_density < 0.6))[0]
            
            # Group into bands
            text_bands = []
            if len(text_rows) > 0:
                current = [text_rows[0]]
                for r in text_rows[1:]:
                    if r - current[-1] <= 4:
                        current.append(r)
                    else:
                        if len(current) > 8:
                            text_bands.append((min(current), max(current)))
                        current = [r]
                if len(current) > 8:
                    text_bands.append((min(current), max(current)))
            
            # Find ALL components
            status.text("Finding all symbols and letters...")
            progress_bar.progress(50)
            
            all_components = []
            
            for band_idx, (by1, by2) in enumerate(text_bands):
                band = legend_bin[by1:by2+1, :]
                
                # Process entire band width
                visited = np.zeros_like(band, dtype=bool)
                
                for y in range(band.shape[0]):
                    for x in range(band.shape[1]):
                        if band[y, x] == 1 and not visited[y, x]:
                            # Flood fill
                            stack = [(y, x)]
                            pixels = []
                            min_x, min_y = x, y
                            max_x, max_y = x, y
                            
                            while stack:
                                cy, cx = stack.pop()
                                if (0 <= cy < band.shape[0] and 
                                    0 <= cx < band.shape[1] and 
                                    band[cy, cx] == 1 and 
                                    not visited[cy, cx]):
                                    
                                    visited[cy, cx] = True
                                    pixels.append((cy, cx))
                                    min_x = min(min_x, cx)
                                    min_y = min(min_y, cy)
                                    max_x = max(max_x, cx)
                                    max_y = max(max_y, cy)
                                    
                                    for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1),
                                                  (cy-1,cx-1),(cy-1,cx+1),(cy+1,cx-1),(cy+1,cx+1)]:
                                        stack.append((ny, nx))
                            
                            comp_w = max_x - min_x + 1
                            comp_h = max_y - min_y + 1
                            comp_area = len(pixels)
                            
                            # Filter size (20-4000 pixels)
                            if 20 < comp_area < 4000 and 8 < comp_w < 200 and 8 < comp_h < 200:
                                
                                # Extract component
                                pad = 3
                                cy1 = max(0, min_y - pad)
                                cy2 = min(band.shape[0], max_y + pad + 1)
                                cx1 = max(0, min_x - pad)
                                cx2 = min(band.shape[1], max_x + pad + 1)
                                
                                comp_img = band[cy1:cy2, cx1:cx2]
                                
                                # Calculate features
                                density = comp_area / (comp_w * comp_h) if comp_w * comp_h > 0 else 0
                                aspect = comp_w / comp_h if comp_h > 0 else 0
                                
                                # Check for horizontal lines
                                h_lines = 0
                                for row in range(comp_img.shape[0]):
                                    if np.sum(comp_img[row, :]) > comp_img.shape[1] * 0.3:
                                        h_lines += 1
                                
                                # Check for vertical lines
                                v_lines = 0
                                for col in range(comp_img.shape[1]):
                                    if np.sum(comp_img[:, col]) > comp_img.shape[0] * 0.3:
                                        v_lines += 1
                                
                                # Check if it's likely a letter (small, dense, tall)
                                is_likely_letter = False
                                
                                # Letters are typically:
                                # - Small area (50-800 pixels)
                                # - High density (30-70%)
                                # - Tall aspect ratio (0.3-1.5)
                                # - Few horizontal lines
                                
                                if (50 < comp_area < 1000 and 
                                    0.2 < density < 0.7 and 
                                    0.2 < aspect < 2.0 and 
                                    h_lines < 3):
                                    is_likely_letter = True
                                
                                # Symbols are typically:
                                # - Larger area (200-4000 pixels)
                                # - Lower density (10-40%)
                                # - More square (0.5-2.0)
                                # - Have horizontal/vertical lines
                                
                                is_likely_symbol = False
                                
                                if (comp_area > 150 and 
                                    density < 0.5 and 
                                    0.4 < aspect < 2.5 and 
                                    (h_lines >= 1 or v_lines >= 1)):
                                    is_likely_symbol = True
                                
                                global_x = cx1
                                global_y = by1 + cy1
                                
                                all_components.append({
                                    'image': comp_img,
                                    'x': global_x,
                                    'y': global_y,
                                    'w': comp_w,
                                    'h': comp_h,
                                    'area': comp_area,
                                    'density': density,
                                    'aspect': aspect,
                                    'h_lines': h_lines,
                                    'v_lines': v_lines,
                                    'is_letter': is_likely_letter,
                                    'is_symbol': is_likely_symbol,
                                    'band': band_idx
                                })
            
            progress_bar.progress(80)
            status.text(f"Found {len(all_components)} components")
            
            # Separate letters and symbols
            letters = [c for c in all_components if c['is_letter'] and not c['is_symbol']]
            symbols = [c for c in all_components if c['is_symbol']]
            unknown = [c for c in all_components if not c['is_letter'] and not c['is_symbol']]
            
            progress_bar.progress(90)
            status.text(f"Letters: {len(letters)} | Symbols: {len(symbols)} | Unknown: {len(unknown)}")
            
            # Mark legend image
            marked_legend = legend_color.copy()
            draw = ImageDraw.Draw(marked_legend)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Mark symbols in GREEN
            for i, sym in enumerate(symbols):
                x1, y1 = sym['x'], sym['y']
                x2, y2 = sym['x'] + sym['w'], sym['y'] + sym['h']
                
                # Green rectangle for symbols
                draw.rectangle([x1-2, y1-2, x2+2, y2+2], outline='lime', width=3)
                
                # Label
                label = f"S{i+1}"
                bbox = draw.textbbox((x1, y1-18), label, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='white')
                draw.text((x1, y1-18), label, fill='green', font=font)
            
            # Mark letters in RED (to show they're excluded)
            for i, let in enumerate(letters[:50]):  # Limit to 50 letters shown
                x1, y1 = let['x'], let['y']
                x2, y2 = let['x'] + let['w'], let['y'] + let['h']
                
                # Red rectangle for letters (excluded)
                draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline='red', width=1)
                draw.text((x1, y1-12), "X", fill='red', font=small_font)
            
            # Mark unknown in YELLOW
            for i, unk in enumerate(unknown):
                x1, y1 = unk['x'], unk['y']
                x2, y2 = unk['x'] + unk['w'], unk['y'] + unk['h']
                
                # Yellow dashed for unknown
                draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline='yellow', width=2)
            
            progress_bar.progress(100)
            status.text("✅ Legend analysis complete!")
            
            # Store in session state
            st.session_state.marked_legend = marked_legend
            st.session_state.all_symbols = symbols
            st.session_state.all_letters = letters
            st.session_state.all_unknown = unknown
            st.session_state.legend_color = legend_color
            st.session_state.legend_bin = legend_bin
            st.session_state.text_bands = text_bands
        
        # Show marked legend
        with col2:
            st.subheader("Marked Legend")
            st.image(st.session_state.marked_legend, use_container_width=True)
            
            # Legend for colors
            st.markdown("""
            - 🟢 **GREEN (S#)** = Potential receptacle symbols
            - 🔴 **RED (X)** = Letters/Alphabets (EXCLUDED)
            - 🟡 **YELLOW** = Unknown (needs review)
            """)
    
    # ===== STEP 3: SELECT RECEPTACLES =====
    if st.session_state.marked_legend is not None:
        st.markdown("---")
        st.header("🔌 Step 3: Select Which Symbols Are Receptacles")
        
        symbols = st.session_state.all_symbols
        
        if symbols:
            st.write(f"Found **{len(symbols)}** potential symbols in legend")
            
            # Show all symbols in a grid
            st.subheader("Click to select receptacle symbols:")
            
            # Create columns for symbols
            cols_per_row = 6
            rows = (len(symbols) + cols_per_row - 1) // cols_per_row
            
            selected = []
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    sym_idx = row * cols_per_row + col_idx
                    if sym_idx < len(symbols):
                        sym = symbols[sym_idx]
                        with cols[col_idx]:
                            # Show symbol image
                            sym_img = Image.fromarray(sym['image'].astype(np.uint8) * 255)
                            st.image(sym_img, width=80)
                            
                            # Show info
                            st.caption(f"S{sym_idx+1}: {sym['w']}×{sym['h']}px")
                            
                            # Checkbox to select
                            is_selected = st.checkbox(
                                f"Receptacle",
                                key=f"sym_{sym_idx}",
                                value=sym.get('is_receptacle', False)
                            )
                            
                            if is_selected:
                                selected.append(sym_idx)
            
            # Confirm selection button
            if st.button("✅ Confirm Selected Receptacles", type="primary", use_container_width=True):
                if selected:
                    st.session_state.confirmed_templates = [symbols[i] for i in selected]
                    st.session_state.legend_confirmed = True
                    st.success(f"✅ **{len(selected)} symbols confirmed as receptacles!**")
                    st.info("Now upload the power plan page to count them.")
                else:
                    st.warning("⚠️ Please select at least one receptacle symbol")
        else:
            st.warning("No symbols found in legend. Try uploading a clearer image.")

# ===== STEP 4: UPLOAD POWER PLAN & COUNT =====
if st.session_state.legend_confirmed:
    st.markdown("---")
    st.header("🔌 Step 4: Upload Power Plan Page")
    st.caption("The floor plan where receptacles will be counted")
    
    power_file = st.file_uploader(
        "Upload power plan image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        key='power_upload'
    )
    
    if power_file:
        power_color = Image.open(power_file).convert('RGB')
        power_gray = power_color.convert('L')
        
        st.subheader("Power Plan")
        st.image(power_color, use_container_width=True)
        
        # ===== STEP 5: COUNT RECEPTACLES =====
        if st.button("🔍 Count Receptacles in Power Plan", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status = st.empty()
            detail = st.empty()
            
            # Preprocess power plan
            status.text("Processing power plan...")
            progress_bar.progress(10)
            
            power_gray = power_gray.filter(ImageFilter.SHARPEN)
            power_arr = np.array(power_gray)
            power_bin = (power_arr < 128).astype(np.uint8)
            
            h_power, w_power = power_bin.shape
            
            # Get confirmed templates
            templates = st.session_state.confirmed_templates
            
            progress_bar.progress(20)
            status.text(f"Searching with {len(templates)} confirmed receptacle templates...")
            
            all_detections = []
            step_size = max(3, min(h_power, w_power) // 80)
            
            for tidx, template in enumerate(templates):
                progress = 20 + int((tidx / len(templates)) * 50)
                progress_bar.progress(progress)
                detail.text(f"Template {tidx+1}/{len(templates)}... Found {len(all_detections)} matches")
                
                timg = template['image']
                th, tw = timg.shape
                
                # Multi-scale search
                for scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
                    new_h = int(th * scale)
                    new_w = int(tw * scale)
                    
                    if new_h < 8 or new_w < 8:
                        continue
                    if new_h > h_power or new_w > w_power:
                        continue
                    
                    tpil = Image.fromarray(timg.astype(np.uint8) * 255)
                    tscaled = np.array(tpil.resize((new_w, new_h), Image.Resampling.LANCZOS)) > 128
                    tscaled = tscaled.astype(np.uint8)
                    
                    black_count = np.sum(tscaled)
                    if black_count < 5:
                        continue
                    
                    for y in range(0, h_power - new_h + 1, step_size):
                        for x in range(0, w_power - new_w + 1, step_size):
                            patch = power_bin[y:y+new_h, x:x+new_w]
                            
                            if patch.shape != tscaled.shape:
                                continue
                            
                            mask = (tscaled == 1)
                            match_black = np.sum(patch[mask])
                            ratio = match_black / black_count
                            
                            if ratio >= match_threshold:
                                all_detections.append({
                                    'x': int(x),
                                    'y': int(y),
                                    'w': new_w,
                                    'h': new_h,
                                    'conf': float(ratio)
                                })
            
            progress_bar.progress(75)
            status.text(f"Raw matches: {len(all_detections)}. Unifying...")
            
            # Remove duplicates (unify nearby detections)
            if all_detections:
                all_detections.sort(key=lambda d: d['conf'], reverse=True)
                
                unified = []
                used = set()
                
                for i, d in enumerate(all_detections):
                    if i in used:
                        continue
                    
                    cluster = [d]
                    used.add(i)
                    
                    for j, d2 in enumerate(all_detections):
                        if j in used:
                            continue
                        
                        ox1 = max(d['x'], d2['x'])
                        oy1 = max(d['y'], d2['y'])
                        ox2 = min(d['x']+d['w'], d2['x']+d2['w'])
                        oy2 = min(d['y']+d['h'], d2['y']+d2['h'])
                        
                        if ox2 > ox1 and oy2 > oy1:
                            overlap = (ox2-ox1) * (oy2-oy1)
                            area = min(d['w']*d['h'], d2['w']*d2['h'])
                            
                            if overlap / area > 0.3:
                                cluster.append(d2)
                                used.add(j)
                    
                    # Average position
                    total_conf = sum(c['conf'] for c in cluster)
                    if total_conf > 0:
                        avg_x = int(sum(c['x'] * c['conf'] for c in cluster) / total_conf)
                        avg_y = int(sum(c['y'] * c['conf'] for c in cluster) / total_conf)
                        avg_w = int(sum(c['w'] * c['conf'] for c in cluster) / total_conf)
                        avg_h = int(sum(c['h'] * c['conf'] for c in cluster) / total_conf)
                    else:
                        avg_x, avg_y = d['x'], d['y']
                        avg_w, avg_h = d['w'], d['h']
                    
                    unified.append({
                        'x': avg_x,
                        'y': avg_y,
                        'w': avg_w,
                        'h': avg_h,
                        'conf': max(c['conf'] for c in cluster)
                    })
                
                detections = unified
            else:
                detections = []
            
            progress_bar.progress(90)
            status.text("Drawing results...")
            
            # Draw results in RED
            result_img = power_color.copy()
            draw = ImageDraw.Draw(result_img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            for i, d in enumerate(detections):
                x1, y1 = d['x'], d['y']
                x2, y2 = d['x'] + d['w'], d['y'] + d['h']
                
                # RED thick border
                draw.rectangle([x1-2, y1-2, x2+2, y2+2], outline='red', width=3)
                
                # Red semi-transparent fill
                overlay = Image.new('RGBA', result_img.size, (0,0,0,0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([x1, y1, x2, y2], fill=(255,0,0,40))
                result_img = Image.alpha_composite(result_img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(result_img)
                
                # Number label
                label = f"R{i+1}"
                bbox = draw.textbbox((x1+3, y1-20), label, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='white')
                draw.text((x1+3, y1-20), label, fill='red', font=font)
            
            progress_bar.progress(100)
            status.text("✅ Complete!")
            
            # Show results
            st.markdown("---")
            
            if detections:
                st.markdown(f"""
                <div style="background:#fff5f5; padding:30px; border-radius:15px; 
                            border:4px solid red; text-align:center; margin:20px 0;">
                    <h2 style="color:#cc0000;">🔌 RECEPTACLES FOUND</h2>
                    <h1 style="color:red; font-size:80px; margin:15px 0;">{len(detections)}</h1>
                    <h3 style="color:#cc0000;">Total Count</h3>
                    <p style="color:#666;">Marked in RED • No alphabets counted</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.image(result_img, caption=f"{len(detections)} Receptacles (RED markers)", use_container_width=True)
                
                # Download
                buf = io.BytesIO()
                result_img.save(buf, format='PNG')
                st.download_button(
                    "📥 Download Result Image",
                    buf.getvalue(),
                    f"receptacles_{len(detections)}_found.png",
                    "image/png",
                    use_container_width=True
                )
                
                # Table
                st.dataframe(
                    [{"Mark": f"R{i+1}", "Position": f"({d['x']},{d['y']})", 
                      "Size": f"{d['w']}×{d['h']}", "Confidence": f"{d['conf']:.0%}"} 
                     for i, d in enumerate(detections)],
                    use_container_width=True
                )
            else:
                st.warning("No receptacles found. Try lowering the match sensitivity.")
