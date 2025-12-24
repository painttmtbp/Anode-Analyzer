import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, time

# ---------------------------------------------------------
# 1. ตั้งค่าหน้า Web App
# ---------------------------------------------------------
st.set_page_config(page_title="Anode Current Monitor v1.0", layout="wide")
st.title("Anode Current Monitor v.1.0 [Updated 24/12/25]")

# ---------------------------------------------------------
# 2. ฟังก์ชันโหลดและรวมข้อมูล 
# ---------------------------------------------------------
@st.cache_data
def load_and_combine_data(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        try:
            temp_df = pd.read_csv(file)
            temp_df.columns = temp_df.columns.str.strip() # Clean column names
            all_dfs.append(temp_df)
        except Exception as e:
            st.error(f"Error reading file {file.name}: {e}")
            
    if not all_dfs:
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # รวม Date+Time -> Timestamp
    combined_df['Timestamp'] = pd.to_datetime(
        combined_df['DATE'] + ' ' + combined_df['TIME'], 
        format='%d/%m/%Y %H:%M:%S', 
        dayfirst=True,
        errors='coerce' 
    )
    
    combined_df.dropna(subset=['Timestamp'], inplace=True)
    combined_df.sort_values(by='Timestamp', inplace=True)
    
    # --- เริ่มต้น: การคำนวณคอลัมน์ใหม่ ---
    
    # 1. Bare Anode B7 (REC.1) - ใช้ค่า B7 โดยตรง
    if 'B7' in combined_df.columns:
        combined_df['Bare Anode B7 (REC.1)'] = combined_df['B7']
    else:
        combined_df['Bare Anode B7 (REC.1)'] = 0

    # 2. Total Left Bare Anode (L6+B4)
    if 'L6' in combined_df.columns and 'B4' in combined_df.columns:
        combined_df['Total Left Bare Anode (L6+B4)'] = combined_df['L6'] + combined_df['B4']
    else:
        combined_df['Total Left Bare Anode (L6+B4)'] = 0

    # 3. Total Right Bare Anode (R6+R8)
    if 'R6' in combined_df.columns and 'R8' in combined_df.columns:
        combined_df['Total Right Bare Anode (R6+R8)'] = combined_df['R6'] + combined_df['R8']
    else:
        combined_df['Total Right Bare Anode (R6+R8)'] = 0

    # 4. Total Bottom Bare Anode (B2+B3+L8+L10+L12)
    bottom_cols = ['B2', 'B3', 'L8', 'L10', 'L12']
    existing_bottom_cols = [col for col in bottom_cols if col in combined_df.columns]
    if existing_bottom_cols:
        combined_df['Total Bottom Bare Anode (B2+B3+L8+L10+L12)'] = combined_df[existing_bottom_cols].sum(axis=1)
    else:
        combined_df['Total Bottom Bare Anode (B2+B3+L8+L10+L12)'] = 0

    # 5. Overall Bare Anode (รวมทุกตัวที่ใช้คำนวณ Bare Anode)
    overall_bare_anode_cols = ['B7', 'L6', 'B4', 'R6', 'R8', 'B2', 'B3', 'L8', 'L10', 'L12']
    existing_overall_cols = [col for col in overall_bare_anode_cols if col in combined_df.columns]
    if existing_overall_cols:
        combined_df['Overall Bare Anode'] = combined_df[existing_overall_cols].sum(axis=1)
    else:
        combined_df['Overall Bare Anode'] = 0

    # --- สิ้นสุด: การคำนวณคอลัมน์ใหม่ ---

    combined_df.set_index('Timestamp', inplace=True)
    
    return combined_df

# ฟังก์ชันคำนวณพื้นที่ใต้กราฟ
def calculate_auc(df, col_name):
    y = df[col_name].values
    x_seconds = (df.index - df.index[0]).total_seconds()
    
    # --- แก้ไขส่วนนี้ (Support NumPy 2.0+) ---
    if hasattr(np, 'trapezoid'):
        area_coulombs = np.trapezoid(y, x_seconds)
    else:
        area_coulombs = np.trapz(y, x_seconds)
    # -------------------------------------

    area_amp_hours = area_coulombs / 3600
    
    # คำนวณค่าเฉลี่ย
    duration_seconds = x_seconds[-1] if len(x_seconds) > 0 else 0
    avg_current = area_coulombs / duration_seconds if duration_seconds > 0 else 0
    
    return area_coulombs, area_amp_hours, avg_current

# --- ฟังก์ชันช่วยสร้าง Trace ตามประเภทกราฟ ---
def create_trace(x_data, y_data, name, chart_type, fixed_size=None):
    common_hover = '%{y:.2f} A'
    
    if chart_type == "Line + Markers (เส้น+จุด)":
        marker_size = fixed_size if fixed_size is not None else (2 if len(x_data) > 100000 else 6)
        return go.Scatter(x=x_data, y=y_data, mode='lines+markers', marker=dict(size=marker_size), name=name, hovertemplate=common_hover)
    
    elif chart_type == "Line (เส้นปกติ)":
        return go.Scatter(x=x_data, y=y_data, mode='lines', name=name, hovertemplate=common_hover)
    
    elif chart_type == "Bar (แท่ง)":
        return go.Bar(x=x_data, y=y_data, name=name, hovertemplate=common_hover)
    
    elif chart_type == "Area (พื้นที่)":
        return go.Scatter(x=x_data, y=y_data, mode='lines', fill='tozeroy', name=name, hovertemplate=common_hover)
    
    elif chart_type == "Scatter (จุดกระจาย)":
        return go.Scatter(x=x_data, y=y_data, mode='markers', marker=dict(size=3), name=name, hovertemplate=common_hover)
    
    else: # Default fallback
        return go.Scatter(x=x_data, y=y_data, mode='lines', name=name)

# ---------------------------------------------------------
# 3. ส่วนอัพโหลดและ Sidebar
# ---------------------------------------------------------
# --- 1. กำหนดสถานะเริ่มต้นของรูปภาพ ---
if 'show_diagram' not in st.session_state:
    st.session_state.show_diagram = False

# --- 2. ฟังก์ชันสำหรับสลับสถานะ ---
def toggle_diagram():
    st.session_state.show_diagram = not st.session_state.show_diagram
    
# --- 3. ฟังก์ชัน Reset Date/Time Filter (ใช้ session_state โดยตรงและลบ st.rerun) ---
def reset_date_filter():
    # ลบ keys ของ widget ออกจาก session_state (เพื่อให้กลับไปใช้ค่าเริ่มต้นที่ value=...)
    if 'start_date_picker' in st.session_state:
        del st.session_state.start_date_picker
    if 'start_time_picker' in st.session_state:
        del st.session_state.start_time_picker
    if 'end_date_picker' in st.session_state:
        del st.session_state.end_date_picker
    if 'end_time_picker' in st.session_state:
        del st.session_state.end_time_picker
        
    # ลบ key ของ Timeframe และ Chart Type เพื่อให้กลับไปใช้ index=0
    if 'timeframe_selector' in st.session_state:
        del st.session_state.timeframe_selector
    if 'chart_type_selector' in st.session_state:
        del st.session_state.chart_type_selector
    
    # NOTE: ไม่ต้องมี st.rerun() ที่นี่ เพราะการเปลี่ยน st.session_state จะสั่ง rerun โดยอัตโนมัติ

    
with st.sidebar:
    st.header("📂 Import .CSV file")
    uploaded_files = st.file_uploader("Mutiple files are acceptable",
        type=['csv'], 
        accept_multiple_files=True
    )
    
    
    st.divider()

    
    if uploaded_files:
        df = load_and_combine_data(uploaded_files)
        
        if df is not None and not df.empty:
            # --- กำหนดกลุ่มคอลัมน์คงที่ (Fixed Total Columns) ---
            all_numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
            
            # Original Fixed Total Columns (Total REC, Total Left/Bottom/Right)
            original_fixed_cols = [col for col in all_numeric_cols if 'Total' in col and 'Bare Anode' not in col] 
            
            # New Fixed Bare Anode Columns (ที่เพิ่งคำนวณเพิ่ม)
            new_fixed_cols = [
                'Bare Anode B7 (REC.1)', 
                'Total Left Bare Anode (L6+B4)', 
                'Total Right Bare Anode (R6+R8)', 
                'Total Bottom Bare Anode (B2+B3+L8+L10+L12)', 
                'Overall Bare Anode'
            ]
            
            # รวมคอลัมน์ Fixed ทั้งหมด
            FIXED_TOTAL_COLS = [col for col in original_fixed_cols + new_fixed_cols if col in all_numeric_cols]
            
            # คอลัมน์ที่ผู้ใช้เลือกได้ (คือทุกคอลัมน์ที่ไม่ใช่คอลัมน์ Total/Bare Anode Total)
            selectable_cols = [col for col in all_numeric_cols if col not in FIXED_TOTAL_COLS and col not in ['DATE', 'TIME', 'Enable Alarm Rec.1', 'Enable Alarm Rec.2']]
            
            # --- Sidebar Settings ---
            st.header("⚙️ Chart Settings")
            
            # 3.0 เลือกโหมดการแสดงผล (Overlay/Stacked)
            st.subheader("1. โหมดการแสดงผล")
            view_mode = st.radio(
                "View Mode:",
                options=["Overlay (ซ้อนกัน)", "Stacked (แยกชั้น)"],
                index=0
            )

            # --- NEW: 3.0.1 เลือกประเภทกราฟ (Chart Type) ---
            st.subheader("2. ประเภทกราฟ (Chart Type)")
            chart_type_options = [
                "Line (เส้นปกติ)", 
                "Line + Markers (เส้น+จุด)", 
                "Bar (แท่ง)",
                "Area (พื้นที่)", 
            ]
            # *** เพิ่ม key ***
            selected_chart_type = st.selectbox("เลือกรูปแบบ:", chart_type_options, index=0, key='chart_type_selector') 

            # 3.1 เลือก Timeframe
            st.subheader("3. ความละเอียดเวลา")
            interval_options = {
                "5s (Default)": None,
                "10s (avg)": "10S", # เอาข้อมูล 5วิ 2 จุดมาเฉลี่ย
                "30s (avg)": "30S",
                "1 hr (avg)": "1H",
                "1 day (avg)": "1D",
                "1 week (avg)": "1W",
                "1 month (avg)": "1MS",
                "1s (Special)": "1S"
            }
            # *** เพิ่ม key ***
            selected_interval_label = st.selectbox(
                "Timeframe:", 
                list(interval_options.keys()),
                index=0,
                key='timeframe_selector'
            )
            selected_resample_rule = interval_options[selected_interval_label]

            # 3.2 เลือก Parameter (ใช้ selectable_cols ที่กรองแล้ว)
            st.subheader("4. เลือกข้อมูล (Modules)")
            
            # กำหนดค่า default ที่ดีกว่า: เลือก L1 และ R1 ถ้ามี
            #default_selection = [c for c in ['L1', 'R1'] if c in selectable_cols]
            
            selected_non_total_cols = st.multiselect(
                "ตัวแปรที่ต้องการพลอต:",
                options=selectable_cols,
                #default=default_selection if default_selection else selectable_cols[0:1]
            )

            # 3.3 Date/Time Picker
            st.subheader("5. กรองช่วงเวลา (Filter)")
            
            min_dt = df.index.min()
            max_dt = df.index.max()
            
            # กำหนดค่าเริ่มต้น/ปัจจุบันของ Date/Time Picker
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🟢 Start")
                # ใช้ keys และกำหนด value เป็น min_dt/max_dt เพื่อให้เป็นค่า Default เมื่อถูก Reset
                start_d = st.date_input("Start Date", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key='start_date_picker')
                start_t = st.time_input("Start Time", value=min_dt.time(), step=60, key='start_time_picker')
            
            with col2:
                st.markdown("🔴 End")
                # ใช้ keys และกำหนด value เป็น min_dt/max_dt เพื่อให้เป็นค่า Default เมื่อถูก Reset
                end_d = st.date_input("End Date", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key='end_date_picker')
                end_t = st.time_input("End Time", value=max_dt.time(), step=60, key='end_time_picker')

            # --- ปุ่ม Reset Filter ---
            st.button("🔄 Reset Date/Time Filter", on_click=reset_date_filter, width='stretch')

            start_date = pd.to_datetime(f"{start_d} {start_t}")
            end_date = pd.to_datetime(f"{end_d} {end_t}")

            if start_date > end_date:
                st.error("⚠️ เวลาเริ่มต้นต้องน้อยกว่าเวลาสิ้นสุด")
                st.stop()
        else:
            st.error("Error reading data.")
            st.stop()
    else:
        st.info("Please upload CSV files.")
        st.stop()
        
    # --- 3. เพิ่มปุ่มควบคุมรูปภาพ ---
    st.divider()
    st.header("Bath Layout")
    
    button_label = "ซ่อน Anode Diagram" if st.session_state.show_diagram else "แสดง Anode Diagram"
    st.button(
        button_label, 
        on_click=toggle_diagram, 
        width='stretch'
    )
    
    # --- 4. แสดงรูปภาพเมื่อสถานะเป็น True ---
    if st.session_state.show_diagram:
        try:
            # ใช้ Anode_Layout.jpg ตามที่ผู้ใช้ระบุ
            st.image(
                'Anode_Layout.jpg', 
                caption="Layout", 
                width='stretch' 
            )
        except FileNotFoundError:
            st.warning("⚠️ ไม่พบไฟล์")


# ---------------------------------------------------------
# 4. Data Processing (ย้ายมาไว้ข้างนอก sidebar)
# ---------------------------------------------------------
if uploaded_files and df is not None and not df.empty:

    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_raw_df = df.loc[mask]

    if filtered_raw_df.empty:
        st.warning("⚠️ ไม่พบข้อมูลในช่วงเวลาที่เลือก")
        st.stop()
    
    # รวมคอลัมน์ที่จะใช้พลอตทั้งหมด
    all_plotting_cols = list(set(selected_non_total_cols + FIXED_TOTAL_COLS))

    plot_df = filtered_raw_df[all_plotting_cols]
    if selected_resample_rule:
        plot_df = plot_df.resample(selected_resample_rule).mean()

    # ---------------------------------------------------------
    # 5. Summary & Plotting (จัดเรียงใหม่ตามคำขอ)
    # ---------------------------------------------------------

    # Config Axis Format
    x_axis_format = {}
    if selected_interval_label == "1 day":
        x_axis_format = dict(tickformat="%d %b", dtick="D1") 
    elif selected_interval_label == "1 month":
        x_axis_format = dict(tickformat="%b '%y", dtick="M1")
    elif selected_interval_label == "1 year":
        x_axis_format = dict(tickformat="%Y", dtick="M12")
    elif selected_interval_label == "5s (Original)":
         x_axis_format = dict(tickformat="%H:%M:%S")
    else:
        x_axis_format = dict(tickformat="%d/%m %H:%M")

    # *********************************************************
    # ******* NEW LAYOUT STEP 1: Overall Summary Table ******
    # *********************************************************
    
    if FIXED_TOTAL_COLS:
        st.subheader("Overall Summary Table") 
        
        # จัดเรียงคอลัมน์ Fixed ใหม่
        total_rec_cols = [col for col in FIXED_TOTAL_COLS if 'REC' in col or col == 'Total']
        bare_anode_cols = [col for col in FIXED_TOTAL_COLS if 'Bare Anode' in col]
        other_total_cols = [col for col in FIXED_TOTAL_COLS if col not in total_rec_cols and col not in bare_anode_cols]
        sorted_fixed_cols = total_rec_cols + bare_anode_cols + other_total_cols
        
        summary_data = []
        for col in sorted_fixed_cols:
            coulombs, amp_hours, avg_current = calculate_auc(filtered_raw_df, col) # เพิ่ม avg_current
            min_val = filtered_raw_df[col].min()
            max_val = filtered_raw_df[col].max()
            
            summary_data.append({
                'Parameter': col,
                'Total Charge (Ah)': f"{amp_hours:,.2f}",
                'Total Charge (C)': f"{coulombs:,.0f}",
                'Avg Current (A)': f"{avg_current:,.2f}", # เพิ่ม Avg Current
                'Min Current (A)': f"{min_val:,.0f}",
                'Max Current (A)': f"{max_val:,.0f}"
            })

        summary_df = pd.DataFrame(summary_data).set_index('Parameter')
        
        st.dataframe(summary_df, width='stretch')
        st.divider()


    # *********************************************************
    # ******* NEW LAYOUT STEP 2: Fixed Total Analysis *****
    # *********************************************************
    
    if FIXED_TOTAL_COLS: 
        
        # ใช้ st.columns เพื่อจัด subheader
        col_h1, col_h2 = st.columns([0.7, 0.3])
        with col_h1:
            col_h1.subheader("Overall Summary")
        
        # --- 1. กำหนดกลุ่มใหม่ (ใช้ซ้ำ) ---
        group1_cols = [col for col in ['Total', 'Total REC.1', 'Total REC.2'] if col in FIXED_TOTAL_COLS]
        group1_title = "Overall Bath (By Rectifier Zone)"
        group2_cols = [col for col in ['Total Left', 'Total Right', 'Total Bottom'] if col in FIXED_TOTAL_COLS]
        group2_title = "Overall Bath (By Side)"
        group3_cols = [col for col in ['Total Left 1', 'Total Bottom 1', 'Total Rigth 1'] if col in FIXED_TOTAL_COLS]
        group3_title = "REC1 Zone"
        group4_cols = [col for col in ['Total Left 2', 'Total Bottom 2', 'Total Rigth 2'] if col in FIXED_TOTAL_COLS]
        group4_title = "REC2 Zone"
        group5_cols = [col for col in new_fixed_cols if col in FIXED_TOTAL_COLS]
        group5_title = "Bare Anode Totals"
        
        fixed_groups = []
        if group1_cols: fixed_groups.append((group1_title, group1_cols))
        if group2_cols: fixed_groups.append((group2_title, group2_cols))
        if group3_cols: fixed_groups.append((group3_title, group3_cols))
        if group4_cols: fixed_groups.append((group4_title, group4_cols))
        if group5_cols: fixed_groups.append((group5_title, group5_cols))

        # --- 2. สร้าง Subplots ---
        num_total_groups = len(fixed_groups)
        subplot_titles = [title for title, _ in fixed_groups]

        fig_total = make_subplots(
            rows=num_total_groups, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )

        # --- 3. เพิ่ม Trace ในแต่ละกลุ่ม (Always Overlay) ---
        for i, (group_title, group_cols) in enumerate(fixed_groups):
            row_index = i + 1
            
            for col in group_cols:
                total_chart_type = "Line (เส้นปกติ)" 
                trace = create_trace(plot_df.index, plot_df[col], col, total_chart_type, fixed_size=None) 
                fig_total.add_trace(trace, row=row_index, col=1)
            
            fig_total.update_yaxes(title_text="Current (Ampere)", row=row_index, col=1)
        
        # --- 4. การตั้งค่า Layout และแกน X ---

        fig_total.update_xaxes(
            showticklabels=True,  
            ticks='outside',
            tickangle=45,          
            **x_axis_format,       
            row = list(range(1, num_total_groups + 1)), 
            col = 1
        )
        
        total_height_fixed = 300 * num_total_groups 
        
        fig_total.update_layout(
            height=total_height_fixed,
            hovermode="x unified", 
            xaxis=dict(
                rangeslider=dict(visible=False), 
                type="date",
                **x_axis_format
            ),
            showlegend=True, 
            title=""
        )
        
        st.plotly_chart(fig_total, width='stretch')
    else:
        st.warning("⚠️ ไม่พบข้อมูล Total หรือ Bare Anode ในไฟล์ที่อัปโหลด")

    st.divider()


    # *********************************************************
    # ******* NEW LAYOUT STEP 3: Module Summary ***********
    # *********************************************************

    if selected_non_total_cols: 
        st.subheader("Module Summary")
        
        # ปรับการแสดงผล Metric ให้เป็น Columns
        num_cols = len(selected_non_total_cols)
        summary_col_blocks = st.columns(num_cols)
        
        for idx, col in enumerate(selected_non_total_cols):
            coulombs, amp_hours, avg_current = calculate_auc(filtered_raw_df, col)
            min_val = filtered_raw_df[col].min()
            max_val = filtered_raw_df[col].max()
            
            with summary_col_blocks[idx]:
                st.markdown(f"**{col}**") 
                st.metric(
                    label=f"Total Charge (Ah)",
                    value=f"{amp_hours:,.2f} Ah",
                    delta=f"{coulombs:,.0f} C",
                    delta_color="off"
                )
                st.metric(
                    label=f"Average Current",
                    value=f"{avg_current:,.2f} A", # เพิ่ม Avg
                    delta_color="off"
                )
                st.metric(
                    label=f"Minimum Current",
                    value=f"{min_val:,.0f} A",
                    delta_color="off"
                )
                st.metric(
                    label=f"Maximum Current",
                    value=f"{max_val:,.0f} A",
                    delta_color="off"
                )
    else:
        st.info("👈 ไม่ได้เลือก Modules ย่อย")
    
    st.divider()


    # *********************************************************
    # ******* NEW LAYOUT STEP 4: Trend Analysis (Modules) *
    # *********************************************************
    
    # กราฟ 6.2 จะแสดงก็ต่อเมื่อมีการเลือก Module
    if selected_non_total_cols: 
        
        # ใช้ st.columns เพื่อจัด subheader
        col_h1_2, col_h2_2 = st.columns([0.7, 0.3])
        with col_h1_2:
            col_h1_2.subheader(f"📈 Trend Analysis (Modules) - {view_mode}")
        
        if view_mode == "Overlay (ซ้อนกัน)":
            fig_main = go.Figure()
            for col in selected_non_total_cols:
                trace = create_trace(plot_df.index, plot_df[col], col, selected_chart_type)
                fig_main.add_trace(trace)
            
            fig_main.update_layout(
                height=450,
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Current (Ampere)",
                xaxis=dict(
                    rangeslider=dict(visible=True), 
                    type="date",
                    **x_axis_format
                ),
                barmode='group' if selected_chart_type == "Bar (แท่ง)" else None
            )

            st.plotly_chart(fig_main, width='stretch')

        else: # Stacked Mode
            num_vars = len(selected_non_total_cols)
            fig_main = make_subplots(
                rows=num_vars, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=selected_non_total_cols
            )

            for i, col in enumerate(selected_non_total_cols):
                trace = create_trace(plot_df.index, plot_df[col], col, selected_chart_type)
                fig_main.add_trace(trace, row=i+1, col=1)

            total_height = 250 * num_vars
            fig_main.update_layout(
                height=total_height,
                hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=False), 
                    type="date",
                    **x_axis_format
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig_main, width='stretch')
    
    st.divider()