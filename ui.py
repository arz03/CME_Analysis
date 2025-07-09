import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Set Streamlit page configuration
st.set_page_config(
    page_title="CME Analysis Dashboard",
    page_icon=None,  # Removed emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page title and subtitle
# # Page title and info
st.title("CME Analysis Dashboard")
st.markdown("### Coronal Mass Ejection Monitoring and Analysis System")
st.markdown("<p style='font-size:30px;color:gray;'>*Dummy data for visualization*</p>", unsafe_allow_html=True)

# Sidebar navigation info
# st.sidebar.title("Navigation")
# st.sidebar.markdown("Use the tabs below to explore different sections of the CME analysis.")

# Load CME dataset (dummy data, replace with real source)
# Generate dummy CME data for 2025
@st.cache_data
def load_cme_data():
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    np.random.seed(42)
    data = pd.DataFrame({
        'date': np.random.choice(dates, 200),
        'speed': np.random.normal(400, 150, 200),
        'angle': np.random.uniform(0, 360, 200),
        'mass': np.random.lognormal(15, 1, 200),
        'energy': np.random.lognormal(30, 1, 200),
        'halo_event': np.random.choice([True, False], 200, p=[0.3, 0.7]),
        'severity': np.random.choice(['Low', 'Medium', 'High'], 200, p=[0.5, 0.3, 0.2])
    })
    return data.sort_values('date')

# Load the CME data
cme_data = load_cme_data()

# Create main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Warnings", 
    "About Us", 
    "2D Graphs", 
    "3D Animations", 
    "AskBot", 
    "Halo CME Catalogue", 
    "APIs"
])

# ---------------------------- Tab 1: Warnings ----------------------------
with tab1:
    st.header("CME Warnings & Alerts")

    st.subheader("Active Warnings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Severity Events", "3", "↑1")
    with col2:
        st.metric("Medium Severity Events", "7", "↓2")
    with col3:
        st.metric("Active Halo Events", "2", "→0")

    st.subheader("Detailed CME Data")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.multiselect("Filter by Severity", ['Low', 'Medium', 'High'], ['Medium', 'High'])
    with col2:
        halo_filter = st.checkbox("Show only Halo events", False)

    # Apply filters to data
    filtered_data = cme_data.copy()
    if severity_filter:
        filtered_data = filtered_data[filtered_data['severity'].isin(severity_filter)]
    if halo_filter:
        filtered_data = filtered_data[filtered_data['halo_event'] == True]

    # Display filtered table
    st.dataframe(
        filtered_data[['date', 'speed', 'mass', 'energy', 'severity', 'halo_event']],
        use_container_width=True,
        height=400
    )

# ---------------------------- Tab 2: About Us ----------------------------
with tab2:
    st.header("About Us")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## Team Neutron Busters

        Our team specializes in monitoring and analyzing Coronal Mass Ejections (CMEs)
        to provide early warning systems for space weather events that can affect:

        - Satellite operations
        - GPS navigation systems
        - Power grids
        - Communication networks
        - Space missions

        ### Our Mission
        To develop advanced predictive models and real-time monitoring systems
        for space weather phenomena, focusing on CME events and their impact on Earth's magnetosphere.

        ### Research Focus
        - Machine learning for CME prediction
        - Real-time data processing
        - Impact assessment models
        - Early warning system development
        """)

# ---------------------------- Tab 3: 2D Graphs ----------------------------
with tab3:
    st.header("2D Graphs & Analysis")

    graph_tab1, graph_tab2, graph_tab3 = st.tabs(["Date Comparisons", "Halo CME Data", "Time Duration Analysis"])

    # Date comparisons
    with graph_tab1:
        st.subheader("CME Event Comparison by Date")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2024-11-15"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2025-12-31"))

        date_filtered = cme_data[
            (cme_data['date'] >= pd.Timestamp(start_date)) &
            (cme_data['date'] <= pd.Timestamp(end_date))
        ]

        # Plot: Speed over time
        fig1 = px.scatter(date_filtered, x='date', y='speed', color='severity', size='mass',
                          title="CME Speed Over Time", labels={'speed': 'Speed (km/s)', 'date': 'Date'})
        st.plotly_chart(fig1, use_container_width=True)

        # Plot: Mass vs energy
        fig2 = px.scatter(date_filtered, x='mass', y='energy', color='halo_event',
                          title="CME Mass vs Energy Correlation")
        st.plotly_chart(fig2, use_container_width=True)

    # Halo CME data
    with graph_tab2:
        st.subheader("Halo CME Data Viewing")
        halo_data = cme_data[cme_data['halo_event'] == True]

        if len(halo_data) > 0:
            fig3 = px.scatter(halo_data, x='date', y='severity', color='speed', size='mass',
                              title="Halo CME Events Over Time")
            st.plotly_chart(fig3, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Halo Events", len(halo_data))
            with col2:
                st.metric("Average Speed", f"{halo_data['speed'].mean():.0f} km/s")
            with col3:
                st.metric("High Severity Halo Events", len(halo_data[halo_data['severity'] == 'High']))
        else:
            st.warning("No Halo events found in the selected data range.")

    # Time-based analysis
    with graph_tab3:
        st.subheader("CME Occurrence Between Time Durations")

        cme_data_copy = cme_data.copy()
        cme_data_copy['year_month'] = cme_data_copy['date'].dt.to_period('M').astype(str)
        monthly_counts = cme_data_copy.groupby('year_month').size()

        fig4 = px.bar(x=monthly_counts.index, y=monthly_counts.values,
                      title="CME Events per Month",
                      labels={'x': 'Month', 'y': 'Number of Events'})
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

        # Severity distribution per month
        severity_time_data = []
        for month in cme_data_copy['year_month'].unique():
            month_data = cme_data_copy[cme_data_copy['year_month'] == month]
            for severity in ['Low', 'Medium', 'High']:
                severity_time_data.append({
                    'Month': month,
                    'Severity': severity,
                    'Count': (month_data['severity'] == severity).sum()
                })

        severity_df = pd.DataFrame(severity_time_data)
        fig5 = px.bar(severity_df, x='Month', y='Count', color='Severity',
                      title="CME Severity Distribution Over Time")
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)

# ---------------------------- Tab 4: 3D Animations ----------------------------
with tab4:
    st.header("3D Animations")

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.time_input("Animation Start Time", value=datetime.now().time())
    with col2:
        end_time = st.time_input("Animation End Time", value=datetime.now().time())

    st.subheader("3D CME Propagation Animation")

    # 3D CME visualization
    # fig_3d = go.Figure(data=[go.Scatter3d(
    #     x=cme_data['speed'] * np.cos(np.radians(cme_data['angle'])),
    #     y=cme_data['speed'] * np.sin(np.radians(cme_data['angle'])),
    #     z=cme_data['mass'],
    #     mode='markers',
    #     marker=dict(
    #         size=cme_data['energy'] / 1e30,
    #         color=cme_data['speed'],
    #         colorscale='Viridis',
    #         showscale=True,
    #         colorbar=dict(title="Speed (km/s)")
    #     ),
    #     text=[f"Date: {d}<br>Speed: {s:.0f} km/s<br>Mass: {m:.2e} kg" for d, s, m in zip(cme_data['date'], cme_data['speed'], cme_data['mass'])],
    #     hovertemplate='%{text}<extra></extra>'
    # )])

    # fig_3d.update_layout(
    #     title="3D CME Visualization",
    #     scene=dict(
    #         xaxis_title="X Velocity Component",
    #         yaxis_title="Y Velocity Component",
    #         zaxis_title="Mass (kg)"
    #     )
    # )
    # st.plotly_chart(fig_3d, use_container_width=True)

    # Animation using Matplotlib and GIF
    if st.button("Start Animation", key="matplotlib_animation_button"):
        with st.spinner("Creating animation..."):
            import matplotlib.pyplot as plt
            from PIL import Image
            import io
            import os
            import tempfile

            def make_frame(ind, temp_dir_path, title, cme_data):
                # Use your 3D plotting logic here, adjusted for Matplotlib
                fig = plt.figure(figsize=(8, 6))  # Adjust figure size as needed
                ax = fig.add_subplot(111, projection='3d')

                # Update positions or other attributes based on the frame index (ind)
                # Here we simulate a change in size and color for animation
                sizes = cme_data['energy'] / 1e30 * (ind / 50)  # Example: scale sizes
                colors = plt.cm.viridis(cme_data['speed'] / cme_data['speed'].max())  # Example: colormap based on speed

                # Plot the 3D scatter plot for this frame
                scatter = ax.scatter(
                    cme_data['speed'] * np.cos(np.radians(cme_data['angle'])),
                    cme_data['speed'] * np.sin(np.radians(cme_data['angle'])),
                    cme_data['mass'],
                    s=sizes,  # Sizes for animation
                    c=cme_data['speed'],  # Colors based on speed
                    cmap='viridis'
                )

                # Set labels and title
                ax.set_xlabel("X Velocity Component")
                ax.set_ylabel("Y Velocity Component")
                ax.set_zlabel("Mass (kg)")
                ax.set_title(f"CME Animation") # More generic title
                ax.text2D(0.05, 0.95, f"Frame: {ind}", transform=ax.transAxes, fontsize=12, verticalalignment='top') # Use text2D for 3D plots

                # Save the frame as a PNG image
                file_path = os.path.join(temp_dir_path, f"{title}_frame_{ind}.png")
                fig.savefig(file_path)
                plt.close(fig)
                return file_path

            def create_gif_animation(paths, duration=200):
                frames = [Image.open(path) for path in paths]
                gif_buffer = io.BytesIO()
                frames[0].save(
                    gif_buffer, format="GIF", save_all=True, append_images=frames[1:],
                    duration=duration, loop=0
                )
                gif_buffer.seek(0)
                return gif_buffer

            # Generate frames
            with tempfile.TemporaryDirectory() as temp_dir_path:
                title = "cme_animation"
                num_frames = 50
                paths_to_fig = []
                for i in range(num_frames + 1):
                    path = make_frame(i, temp_dir_path, title, cme_data)
                    paths_to_fig.append(path)

                # Create GIF
                gif_buffer = create_gif_animation(paths_to_fig, duration=100)  # Adjust duration for speed

                # Display GIF in Streamlit
                st.image(gif_buffer.getvalue(), caption="CME Propagation Animation", use_column_width=True)

# ---------------------------- Tab 5: AskBot ----------------------------
with tab5:
    st.header("CME Analysis AskBot")

    st.markdown("Ask questions about CME data and get AI-powered insights!")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your CME Analysis assistant. Ask me anything about coronal mass ejections, space weather, or the data in our system."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about CME data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = f"Based on our CME data analysis, here's an answer to: {prompt} (This is a placeholder response)."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------------------- Tab 6: Halo CME Catalogue ----------------------------
with tab6:
    st.header("Our Halo CME Catalogue")

    st.subheader("Upcoming CME Event Ratings & Predictions")

    future_events = pd.DataFrame({
        'Predicted_Date': pd.date_range(start='2025-07-10', periods=10, freq='D'),
        'Probability': np.random.uniform(0.1, 0.9, 10),
        'Predicted_Speed': np.random.normal(500, 100, 10),
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High'], 10),
        'Confidence': np.random.uniform(0.6, 0.95, 10)
    })

    st.dataframe(future_events, use_container_width=True)

    risk_counts = future_events['Risk_Level'].value_counts()
    fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                      title="Upcoming CME Risk Distribution")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prediction Accuracy", "87.3%", "↑2.1%")
    with col2:
        st.metric("False Positive Rate", "12.7%", "↓1.3%")
    with col3:
        st.metric("Early Warning Time", "18.4 hrs", "↑0.7 hrs")
    with col4:
        st.metric("Model Confidence", "91.2%", "↑1.8%")

# ---------------------------- Tab 7: APIs ----------------------------
with tab7:
    st.header("APIs & Data Access")

    st.subheader("Available APIs")

    api_info = {
        "CME Data API": {
            "endpoint": "/api/v1/cme-data",
            "methods": ["GET", "POST"],
            "description": "Retrieve historical CME event data",
            "parameters": "date_range, severity, halo_filter"
        },
        "Prediction API": {
            "endpoint": "/api/v1/predictions",
            "methods": ["GET"],
            "description": "Get CME predictions and forecasts",
            "parameters": "forecast_days, confidence_threshold"
        },
        "Real-time Alerts API": {
            "endpoint": "/api/v1/alerts",
            "methods": ["GET", "POST"],
            "description": "Subscribe to real-time CME alerts",
            "parameters": "alert_type, notification_method"
        }
    }

    for api_name, details in api_info.items():
        with st.expander(f"{api_name}"):
            st.code(f"""
Endpoint: {details['endpoint']}
Methods: {', '.join(details['methods'])}
Description: {details['description']}
Parameters: {details['parameters']}

Example Request:
curl -X GET "https://api.cme-analysis.org{details['endpoint']}?date_range=2024-01-01,2024-12-31"
            """, language="bash")

    st.subheader("API Key Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate New API Key"):
            st.success("New API key generated: `cme_api_key_123456789`")
    with col2:
        if st.button("View Usage Statistics"):
            st.info("API calls this month: 1,247\nRemaining quota: 8,753")

    st.subheader("Processed Data Downloads")
    download_options = {
        "Complete CME Dataset (CSV)": "cme_complete_dataset.csv",
        "Halo Events Only (JSON)": "halo_events.json",
        "Prediction Model Outputs (CSV)": "predictions.csv",
        "API Documentation (PDF)": "api_docs.pdf"
    }

    for desc, filename in download_options.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(desc)
        with col2:
            st.download_button(
                label="Download",
                data="Sample data content",  # Replace with actual file data
                file_name=filename,
                mime="text/csv" if filename.endswith('.csv') else "application/json"
            )

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.markdown("© 2025 Team Neutron Busters | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
