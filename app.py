import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="Receptacle Counter", page_icon="🔌", layout="wide")

st.title("🔌 Electrical Receptacle Counter")
st.write("Upload images to count electrical receptacles")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Legend Page")
    legend_file = st.file_uploader("Upload legend image", type=['png','jpg','jpeg'])
    if legend_file:
        st.image(legend_file, use_container_width=True)

with col2:
    st.subheader("Step 2: Power Plan Page")
    power_file = st.file_uploader("Upload power plan image", type=['png','jpg','jpeg'])
    if power_file:
        st.image(power_file, use_container_width=True)

if legend_file and power_file:
    st.markdown("---")
    if st.button("🔍 Count Receptacles", type="primary", use_container_width=True):
        
        with st.spinner("Processing... Please wait"):
            
            # Load images
            legend = Image.open(legend_file).convert('L')
            power = Image.open(power_file).convert('L')
            power_color = Image.open(power_file).convert('RGB')
            
            # Make images sharper
            legend = legend.filter(ImageFilter.SHARPEN)
            power = power.filter(ImageFilter.SHARPEN)
            
            # Convert to black and white
            legend_bin = legend.point(lambda x: 0 if x < 128 else 255, '1')
            power_bin = power.point(lambda x: 0 if x < 128 else 255, '1')
            
            # Convert to numpy arrays
            legend_arr = np.array(legend_bin)
            power_arr = np.array(power_bin)
            
            # Get legend dimensions
            h, w = legend_arr.shape
            
            # Look at left side of legend (where symbols usually are)
            left_region = legend_arr[:, :int(w * 0.4)]
            
            # Find symbols in legend
            visited = np.zeros_like(left_region, dtype=bool)
            templates = []
            
            progress_bar = st.progress(0)
            status = st.empty()
            status.text("Extracting symbols from legend...")
            
            # Find connected black pixels (symbols)
            for y in range(left_region.shape[0]):
                for x in range(left_region.shape[1]):
                    if left_region[y, x] == 0 and not visited[y, x]:
                        # Found a black pixel - flood fill to find whole symbol
                        stack = [(y, x)]
                        pixels = []
                        min_x, min_y = x, y
                        max_x, max_y = x, y
                        
                        while stack:
                            cy, cx = stack.pop()
                            if (0 <= cy < left_region.shape[0] and 
                                0 <= cx < left_region.shape[1] and 
                                left_region[cy, cx] == 0 and 
                                not visited[cy, cx]):
                                
                                visited[cy, cx] = True
                                pixels.append((cy, cx))
                                
                                # Update boundaries
                                min_x = min(min_x, cx)
                                min_y = min(min_y, cy)
                                max_x = max(max_x, cx)
                                max_y = max(max_y, cy)
                                
                                # Check neighbors (up, down, left, right)
                                for ny, nx in [(cy-1,cx), (cy+1,cx), (cy,cx-1), (cy,cx+1)]:
                                    stack.append((ny, nx))
                        
                        # If symbol is right size, save it as template
                        if 100 < len(pixels) < 10000:
                            template = legend_arr[min_y:max_y+1, min_x:max_x+1]
                            templates.append(template)
            
            status.text(f"Found {len(templates)} symbols in legend")
            progress_bar.progress(30)
            
            # Search for symbols in power plan
            detections = []
            
            # How many pixels to jump (bigger = faster but less accurate)
            step = max(5, min(power_arr.shape[0], power_arr.shape[1]) // 50)
            
            for tidx, template in enumerate(templates[:5]):  # Use first 5 templates
                th, tw = template.shape
                
                if th < 10 or tw < 10:
                    continue
                
                # Try different sizes
                for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    new_h = int(th * scale)
                    new_w = int(tw * scale)
                    
                    if new_h < 10 or new_w < 10:
                        continue
                    if new_h > power_arr.shape[0] or new_w > power_arr.shape[1]:
                        continue
                    
                    # Resize template
                    tpil = Image.fromarray(template.astype(np.uint8) * 255)
                    tscaled = np.array(tpil.resize((new_w, new_h))) // 255
                    
                    # Slide template across power plan
                    for y in range(0, power_arr.shape[0] - new_h, step):
                        for x in range(0, power_arr.shape[1] - new_w, step):
                            patch = power_arr[y:y+new_h, x:x+new_w]
                            
                            if patch.shape == tscaled.shape:
                                # Count matching pixels
                                matches = np.sum(patch == tscaled)
                                total = tscaled.size
                                similarity = matches / total
                                
                                if similarity > 0.7:  # 70% match threshold
                                    detections.append({
                                        'x': int(x),
                                        'y': int(y),
                                        'w': new_w,
                                        'h': new_h,
                                        'conf': float(similarity)
                                    })
                
                # Update progress
                progress = 30 + int((tidx + 1) / min(5, len(templates)) * 40)
                progress_bar.progress(progress)
                status.text(f"Searching with template {tidx+1}/{min(5, len(templates))}... Found {len(detections)} matches")
            
            # Remove overlapping detections (keep the best one)
            progress_bar.progress(80)
            status.text("Removing duplicates...")
            
            if detections:
                # Sort by confidence (highest first)
                detections.sort(key=lambda d: d['conf'], reverse=True)
                
                # Keep only non-overlapping detections
                final = [detections[0]]
                
                for d in detections[1:]:
                    # Check if this detection overlaps with any we kept
                    overlap = False
                    for f in final:
                        if (abs(d['x'] - f['x']) < 30 and 
                            abs(d['y'] - f['y']) < 30):
                            overlap = True
                            break
                    
                    if not overlap:
                        final.append(d)
                
                detections = final
            
            progress_bar.progress(100)
            status.text("Done!")
            
            # Show results
            st.markdown("---")
            
            if detections:
                # Success message
                st.success(f"🎉 Found {len(detections)} receptacles!")
                st.markdown(f"### Total Count: {len(detections)}")
                
                # Draw boxes on image
                draw = ImageDraw.Draw(power_color)
                
                for i, d in enumerate(detections[:50]):  # Show up to 50
                    # Green rectangle
                    draw.rectangle(
                        [d['x'], d['y'], d['x']+d['w'], d['y']+d['h']],
                        outline='lime',
                        width=3
                    )
                    
                    # Number label
                    draw.text(
                        (d['x']+2, d['y']-15),
                        f"#{i+1}",
                        fill='lime'
                    )
                
                st.image(power_color, caption=f"Found {len(detections)} receptacles", use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                power_color.save(buf, format='PNG')
                
                st.download_button(
                    label="📥 Download Result Image",
                    data=buf.getvalue(),
                    file_name=f"receptacles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
                
                # Show coordinates
                st.subheader("Detection Details")
                for i, d in enumerate(detections[:10]):  # Show first 10
                    st.write(f"#{i+1}: Position ({d['x']}, {d['y']}) - Confidence: {d['conf']:.1%}")
                
            else:
                st.warning("No receptacles found. Try:")
                st.write("- Make sure legend has clear receptacle symbols")
                st.write("- Use higher quality images")
                st.write("- Symbols should be black on white background")
                st.write("- Legend and power plan should be from same project")
