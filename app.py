import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, time

# ---------------------------------------------------------
# 1. ตั้งค่าหน้า Web App
# ---------------------------------------------------------
st.set_page_config(page_title="EDP Anode Current Analyzer", layout="wide")
st.title("⚡ Anode Current Analyzer")

# ---------------------------------------------------------
# 2. ฟังก์ชันโหลดและรวมข้อมูล
# ---------------------------------------------------------
@st.cache_data
def load_and_combine_data(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        try:
            temp_df = pd.read_csv(file)
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
    combined_df.set_index('Timestamp', inplace=True)
    
    return combined_df

# ฟังก์ชันคำนวณพื้นที่ใต้กราฟ
def calculate_auc(df, col_name):
    y = df[col_name].values
    x_seconds = (df.index - df.index[0]).total_seconds()
    area_coulombs = np.trapz(y, x_seconds)
    area_amp_hours = area_coulombs / 3600
    return area_coulombs, area_amp_hours

# --- ฟังก์ชันช่วยสร้าง Trace ตามประเภทกราฟ ---
def create_trace(x_data, y_data, name, chart_type):
    common_hover = '%{y:.2f} A'
    
    if chart_type == "Line (เส้นปกติ)":
        return go.Scatter(x=x_data, y=y_data, mode='lines', name=name, hovertemplate=common_hover)
    
    elif chart_type == "Line + Markers (เส้น+จุด)":
        # ปรับขนาดจุดอัตโนมัติตามจำนวนข้อมูล
        marker_size = 2 if len(x_data) > 100000 else 6
        return go.Scatter(x=x_data, y=y_data, mode='lines+markers', marker=dict(size=marker_size), name=name, hovertemplate=common_hover)
    
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
with st.sidebar:
    st.header("📂 Data Import")
    uploaded_files = st.file_uploader(
        "อัพโหลดไฟล์ CSV (หลายไฟล์ได้)", 
        type=['csv'], 
        accept_multiple_files=True
    )

if uploaded_files:
    df = load_and_combine_data(uploaded_files)
    
    if df is not None and not df.empty:
        # --- Sidebar Settings ---
        with st.sidebar:
            st.divider()
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
                "Scatter (จุดกระจาย)"
            ]
            selected_chart_type = st.selectbox("เลือกรูปแบบ:", chart_type_options, index=0)

            # 3.1 เลือก Timeframe
            st.subheader("3. ความละเอียดเวลา")
            interval_options = {
                "5s (Original)": None,  
                "1 min": "1T",
                "1 hr": "1H",
                "1 day": "1D",
                "1 week": "1W",
                "1 month": "1MS",       
                "1 year": "1YS"         
            }
            selected_interval_label = st.selectbox(
                "Timeframe:", 
                list(interval_options.keys()),
                index=1 # Default ที่ 1 min
            )
            selected_resample_rule = interval_options[selected_interval_label]

            # 3.2 เลือก Parameter
            st.subheader("4. เลือกข้อมูล (Parameters)")
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
            default_cols = [c for c in ['Total REC.1', 'Total REC.2'] if c in numeric_cols]
            
            selected_cols = st.multiselect(
                "ตัวแปรที่ต้องการพลอต:",
                options=numeric_cols,
                default=default_cols if default_cols else numeric_cols[0:1]
            )

            # 3.3 Date/Time Picker
            st.subheader("5. กรองช่วงเวลา (Filter)")
            
            min_dt = df.index.min()
            max_dt = df.index.max()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🟢 Start")
                start_d = st.date_input("Start Date", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
                start_t = st.time_input("Start Time", value=min_dt.time(), step=60)
            
            with col2:
                st.markdown("🔴 End")
                end_d = st.date_input("End Date", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
                end_t = st.time_input("End Time", value=max_dt.time(), step=60)

            start_date = pd.to_datetime(f"{start_d} {start_t}")
            end_date = pd.to_datetime(f"{end_d} {end_t}")

            if start_date > end_date:
                st.error("⚠️ เวลาเริ่มต้นต้องน้อยกว่าเวลาสิ้นสุด")
                st.stop()

        # ---------------------------------------------------------
        # 4. Data Processing
        # ---------------------------------------------------------
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_raw_df = df.loc[mask]

        if filtered_raw_df.empty:
            st.warning("⚠️ ไม่พบข้อมูลในช่วงเวลาที่เลือก โปรดปรับช่วงเวลาใหม่")
            st.stop()

        if not selected_cols:
            st.warning("👈 เลือก Parameter อย่างน้อย 1 ตัว")
            st.stop()

        plot_df = filtered_raw_df[selected_cols]
        if selected_resample_rule:
            plot_df = plot_df.resample(selected_resample_rule).mean()

        # ---------------------------------------------------------
        # 5. AUC Calculation
        # ---------------------------------------------------------
        st.markdown("### Total Charge Calculation")
        cols = st.columns(len(selected_cols))
        for idx, col in enumerate(selected_cols):
            # คำนวณจาก Raw Data เสมอเพื่อความแม่นยำ
            coulombs, amp_hours = calculate_auc(filtered_raw_df, col)
            with cols[idx]:
                st.metric(
                    label=f"{col}",
                    value=f"{amp_hours:,.2f} Ah",
                    delta=f"{coulombs:,.0f} C",
                    delta_color="off"
                )
        
        st.divider()

        # ---------------------------------------------------------
        # 6. Plotting Logic
        # ---------------------------------------------------------
        st.subheader(f"📈 Trend Analysis ({view_mode})")

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

        if view_mode == "Overlay (ซ้อนกัน)":
            fig = go.Figure()
            for col in selected_cols:
                # เรียกใช้ Helper Function เพื่อสร้าง Trace ตาม Chart Type
                trace = create_trace(plot_df.index, plot_df[col], col, selected_chart_type)
                fig.add_trace(trace)
            
            fig.update_layout(
                height=600,
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Current (Ampere)",
                xaxis=dict(
                    rangeslider=dict(visible=True), 
                    type="date",
                    **x_axis_format
                ),
                # ถ้าเป็น Bar Chart แบบ Overlay ปรับให้โปร่งแสงนิดนึงจะได้เห็นข้อมูลซ้อนกันได้
                barmode='group' if selected_chart_type == "Bar (แท่ง)" else None
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Stacked Mode (แยกชั้น)
            num_vars = len(selected_cols)
            fig = make_subplots(
                rows=num_vars, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=selected_cols
            )

            for i, col in enumerate(selected_cols):
                trace = create_trace(plot_df.index, plot_df[col], col, selected_chart_type)
                fig.add_trace(trace, row=i+1, col=1)

            total_height = 300 * num_vars
            fig.update_layout(
                height=total_height,
                hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=False), 
                    type="date",
                    **x_axis_format
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Error reading data.")
else:
    st.info("Please upload CSV files.")