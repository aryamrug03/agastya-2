import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Category Analysis Dashboard", layout="wide", page_icon="📊")

# ===== CATEGORY ANALYSIS FUNCTIONS =====

def process_category_data(df):
    """
    Process category-wise data for analysis
    """
    # Clean data
    df = df.dropna(subset=['Category', 'Student Id'])
    
    # Extract parent class (e.g., "6-A" -> "6")
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    df = df[df['Parent_Class'].notna()]
    
    # Standardize Program Types - Comprehensive mapping
    program_type_mapping = {
        'SCB': 'PCMB',
        'SCC': 'PCMB',
        'SCM': 'PCMB',
        'SCP': 'PCMB',
        'E-LOB': 'ELOB',
        'ELOB': 'ELOB',  # Keep ELOB as ELOB
        'E LOB': 'ELOB',  # Handle spacing variations
        'DLC2': 'DLC',  # Handle variations without hyphen
        'DLC': 'DLC'  # Keep DLC as DLC
    }
    
    # Apply mapping with strip to handle any whitespace
    df['Program Type'] = df['Program Type'].str.strip().replace(program_type_mapping)
    
    # Standardize category names
    df['Category'] = df['Category'].str.strip().str.title()

    # Standardize subject names
    df["Subject'] = df['Subject'].str.strip().str.title() 
    
    # Ensure correct answer columns are numeric
    df['Student Correct Answer Pre'] = pd.to_numeric(df['Student Correct Answer Pre'], errors='coerce').fillna(0)
    df['Student Correct Answer Post'] = pd.to_numeric(df['Student Correct Answer Post'], errors='coerce').fillna(0)
    
    return df

def calculate_category_metrics(df, filters=None):
    """
    Calculate category-wise performance metrics
    """
    filtered_df = df.copy()
    
    # Apply filters if provided
    if filters:
        if filters.get('region') and filters['region'] != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == filters['region']]
        if filters.get('program_type') and filters['program_type'] != 'All':
            filtered_df = filtered_df[filtered_df['Program Type'] == filters['program_type']]
        if filters.get('subject') and filters['subject'] != 'All':
            filtered_df = filtered_df[filtered_df['Subject'] == filters['subject']]
        if filters.get('topic') and filters['topic'] != 'All':
            filtered_df = filtered_df[filtered_df['Topic Name'] == filters['topic']]
        if filters.get('class') and filters['class'] != 'All':
            filtered_df = filtered_df[filtered_df['Parent_Class'] == filters['class']]
        if filters.get('difficulty') and filters['difficulty'] != 'All':
            filtered_df = filtered_df[filtered_df['Difficulty'] == filters['difficulty']]
    
    # Calculate metrics by category
    category_stats = filtered_df.groupby('Category').agg({
        'Student Correct Answer Pre': ['sum', 'count'],
        'Student Correct Answer Post': ['sum', 'count']
    }).reset_index()
    
    category_stats.columns = ['Category', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
    
    # Calculate percentages
    category_stats['Pre_Percentage'] = (category_stats['Pre_Correct'] / category_stats['Pre_Total'] * 100).round(2)
    category_stats['Post_Percentage'] = (category_stats['Post_Correct'] / category_stats['Post_Total'] * 100).round(2)
    category_stats['Improvement'] = (category_stats['Post_Percentage'] - category_stats['Pre_Percentage']).round(2)
    
    # Calculate lagging percentage (questions that went from correct to incorrect)
    lagging_data = []
    for category in filtered_df['Category'].unique():
        cat_data = filtered_df[filtered_df['Category'] == category]
        # Students who got it right in pre but wrong in post
        lagging = ((cat_data['Student Correct Answer Pre'] == 1) & 
                  (cat_data['Student Correct Answer Post'] == 0)).sum()
        total = len(cat_data)
        lagging_pct = (lagging / total * 100) if total > 0 else 0
        lagging_data.append({'Category': category, 'Lagging_Percentage': round(lagging_pct, 2)})
    
    lagging_df = pd.DataFrame(lagging_data)
    category_stats = category_stats.merge(lagging_df, on='Category', how='left')
    
    return category_stats

# ===== MAIN APPLICATION =====

st.title("📊 Category-wise Student Performance Analysis")
st.markdown("### Analyze Student Performance Across Question Categories")

# File uploader
st.markdown("#### Upload Category Analysis Data")
category_file = st.file_uploader("Upload Excel File with Category Data", type=['xlsx', 'xls'])

# Check if file is uploaded
if category_file is None:
    st.info("👆 Please upload your category analysis Excel file to begin")
    st.markdown("---")
    st.subheader("📋 File Requirements")
    
    st.markdown("""
    **Your Excel file must include these columns:**
    
    **Required Columns:**
    - `Category` - Question category (Factual, Analytical, Application Base, Conceptual)
    - `Student Correct Answer Pre` - Binary (0 or 1) indicating if student answered correctly in pre-test
    - `Student Correct Answer Post` - Binary (0 or 1) indicating if student answered correctly in post-test
    - `Question No` - Question identifier (e.g., Q1, Q2)
    - `Difficulty` - Question difficulty level (EASY, MEDIUM, HARD)
    
    **Additional Required Columns for Filtering:**
    - `Region` - Geographic region
    - `Program Type` - Type of program
    - `Subject` - Subject name (e.g., MATH, SCIENCE)
    - `Topic Name` - Specific topic within subject
    - `Class` - Student's class (e.g., 6-A, 7-B)
    - `Student Id` - Unique student identifier
    
    **Optional but Recommended:**
    - `Student Name` - Student's name
    - `School Name` - School name
    - `State` - State name
    - `Instructor Name` - Instructor's name
    - `Date` - Date of assessment
    - `Question` - Full question text
    - `Correct Answer` - The correct answer option
    - `Student Answer Pre` - Student's pre-test answer
    - `Student Answer Post` - Student's post-test answer
    
    ---
    
    **Analysis Features:**
    - ✅ Category-wise performance comparison (Pre vs Post)
    - ✅ Improvement tracking by category
    - ✅ Lagging percentage identification
    - ✅ Multi-dimensional filtering (Region, Program Type, Subject, Topic, Grade, Difficulty)
    - ✅ Deep-dive analysis for individual categories
    - ✅ Performance breakdowns by Region, Grade, Difficulty, and Topic
    """)
    st.stop()

# Load and process data
try:
    with st.spinner("Loading and processing data..."):
        category_df = pd.read_excel(category_file)
        
        # Validate required columns
        required_columns = ['Category', 'Student Correct Answer Pre', 'Student Correct Answer Post', 
                          'Question No', 'Difficulty', 'Region', 'Program Type', 'Subject', 
                          'Topic Name', 'Class', 'Student Id']
        
        missing_columns = [col for col in required_columns if col not in category_df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info(f"Available columns in your file: {', '.join(category_df.columns.tolist()[:15])}...")
            st.stop()
        
        # Process the data
        category_df = process_category_data(category_df)
        
        st.success(f"✅ Data loaded successfully: {len(category_df)} question attempts analyzed")
        
        # Show data standardization info
        st.info("📝 Program Types have been standardized: SCB/SCC/SCM/SCP → PCMB | E-LOB → ELOB | DLC-2 → DLC")
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Question Attempts", len(category_df))
        with col2:
            st.metric("Categories", category_df['Category'].nunique())
        with col3:
            st.metric("Topics", category_df['Topic Name'].nunique())

except Exception as e:
    st.error(f"Error processing file: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

# Download processed data option
st.markdown("---")
category_excel = category_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Processed Data (CSV)",
    data=category_excel,
    file_name="processed_category_data.csv",
    mime="text/csv"
)

# ===== CATEGORY ANALYSIS =====

st.markdown("---")
st.header("📚 Category-wise Performance Analysis")

# Sidebar filters for category analysis
st.sidebar.header("🔍 Filters")

filters = {}
filters['region'] = st.sidebar.selectbox("Region", ['All'] + sorted(category_df['Region'].unique().tolist()))
filters['program_type'] = st.sidebar.selectbox("Program Type", ['All'] + sorted(category_df['Program Type'].unique().tolist()))
filters['subject'] = st.sidebar.selectbox("Subject", ['All'] + sorted(category_df['Subject'].unique().tolist()))
filters['topic'] = st.sidebar.selectbox("Topic Name", ['All'] + sorted(category_df['Topic Name'].unique().tolist()))
filters['class'] = st.sidebar.selectbox("Grade", ['All'] + sorted(category_df['Parent_Class'].unique().tolist()))
filters['difficulty'] = st.sidebar.selectbox("Difficulty", ['All'] + sorted(category_df['Difficulty'].unique().tolist()))

# Calculate metrics
category_stats = calculate_category_metrics(category_df, filters)

# Display filtered data metrics
st.subheader("📊 Filtered Data Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Categories Analyzed", len(category_stats))
with col2:
    avg_pre = category_stats['Pre_Percentage'].mean()
    st.metric("Avg Pre-Session %", f"{avg_pre:.1f}%")
with col3:
    avg_post = category_stats['Post_Percentage'].mean()
    st.metric("Avg Post-Session %", f"{avg_post:.1f}%")
with col4:
    avg_improvement = category_stats['Improvement'].mean()
    st.metric("Avg Improvement", f"{avg_improvement:.1f}%", delta=f"{avg_improvement:.1f}%")

st.markdown("---")

# Overall category comparison
st.subheader("📊 Overall Category Performance Comparison")

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Pre-Session',
    x=category_stats['Category'],
    y=category_stats['Pre_Percentage'],
    marker_color='#3498db',
    text=[f"{val:.1f}%" for val in category_stats['Pre_Percentage']],
    textposition='outside',
    textfont=dict(size=12)
))

fig.add_trace(go.Bar(
    name='Post-Session',
    x=category_stats['Category'],
    y=category_stats['Post_Percentage'],
    marker_color='#2ecc71',
    text=[f"{val:.1f}%" for val in category_stats['Post_Percentage']],
    textposition='outside',
    textfont=dict(size=12)
))

fig.update_layout(
    title='Category-wise Correct Answer Percentage (Pre vs Post)',
    xaxis_title='Category',
    yaxis_title='Percentage Correct (%)',
    barmode='group',
    height=500,
    plot_bgcolor='#2b2b2b',
    paper_bgcolor='#1e1e1e',
    font=dict(color='white'),
    yaxis=dict(range=[0, 110], gridcolor='#404040'),
    xaxis=dict(gridcolor='#404040')
)

st.plotly_chart(fig, use_container_width=True)

# Improvement and Lagging
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Improvement by Category")
    fig_imp = go.Figure()
    
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in category_stats['Improvement']]
    
    fig_imp.add_trace(go.Bar(
        x=category_stats['Category'],
        y=category_stats['Improvement'],
        marker_color=colors,
        text=[f"{val:+.1f}%" for val in category_stats['Improvement']],
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig_imp.update_layout(
        title='Improvement (Post - Pre)',
        xaxis_title='Category',
        yaxis_title='Improvement (%)',
        height=400,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(gridcolor='#404040', zeroline=True, zerolinecolor='white', zerolinewidth=2),
        xaxis=dict(gridcolor='#404040')
    )
    
    st.plotly_chart(fig_imp, use_container_width=True)

with col2:
    st.subheader("⚠️ Lagging Percentage by Category")
    fig_lag = go.Figure()
    
    fig_lag.add_trace(go.Bar(
        x=category_stats['Category'],
        y=category_stats['Lagging_Percentage'],
        marker_color='#e67e22',
        text=[f"{val:.1f}%" for val in category_stats['Lagging_Percentage']],
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig_lag.update_layout(
        title='Students Who Regressed (Correct → Incorrect)',
        xaxis_title='Category',
        yaxis_title='Lagging (%)',
        height=400,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(gridcolor='#404040'),
        xaxis=dict(gridcolor='#404040')
    )
    
    st.plotly_chart(fig_lag, use_container_width=True)

# Detailed statistics table
st.markdown("---")
st.subheader("📋 Detailed Category Statistics")

display_stats = category_stats.copy()
display_stats = display_stats[['Category', 'Pre_Percentage', 'Post_Percentage', 'Improvement', 'Lagging_Percentage']]
display_stats.columns = ['Category', 'Pre %', 'Post %', 'Improvement %', 'Lagging %']

st.dataframe(display_stats, hide_index=True, use_container_width=True)

# Individual category deep dive
st.markdown("---")
st.subheader("🔍 Individual Category Deep Dive")

selected_category = st.selectbox("Select Category for Detailed Analysis", 
                                 sorted(category_df['Category'].unique()))

cat_data = category_df[category_df['Category'] == selected_category].copy()

# Apply same filters
if filters['region'] != 'All':
    cat_data = cat_data[cat_data['Region'] == filters['region']]
if filters['program_type'] != 'All':
    cat_data = cat_data[cat_data['Program Type'] == filters['program_type']]
if filters['subject'] != 'All':
    cat_data = cat_data[cat_data['Subject'] == filters['subject']]
if filters['topic'] != 'All':
    cat_data = cat_data[cat_data['Topic Name'] == filters['topic']]
if filters['class'] != 'All':
    cat_data = cat_data[cat_data['Parent_Class'] == filters['class']]
if filters['difficulty'] != 'All':
    cat_data = cat_data[cat_data['Difficulty'] == filters['difficulty']]

if len(cat_data) == 0:
    st.warning(f"⚠️ No data available for {selected_category} with the current filters.")
else:
    # Show category-specific metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pre_correct = cat_data['Student Correct Answer Pre'].sum()
        pre_total = len(cat_data)
        pre_pct = (pre_correct / pre_total) * 100 if pre_total > 0 else 0
        st.metric(f"Pre-Session %", f"{pre_pct:.1f}%")
    
    with col2:
        post_correct = cat_data['Student Correct Answer Post'].sum()
        post_total = len(cat_data)
        post_pct = (post_correct / post_total) * 100 if post_total > 0 else 0
        st.metric(f"Post-Session %", f"{post_pct:.1f}%")
    
    with col3:
        improvement = post_pct - pre_pct
        st.metric(f"Improvement", f"{improvement:+.1f}%", delta=f"{improvement:+.1f}%")
    
    with col4:
        st.metric(f"Question Attempts", len(cat_data))
    
    # Sub-tabs for different breakdowns
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 By Region", "📚 By Grade", "📊 By Program Type", "📖 By Topic", "⚡ By Difficulty"])
    
    with tab1:
        st.markdown(f"#### {selected_category} - Performance by Region")
        region_stats = cat_data.groupby('Region').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        region_stats.columns = ['Region', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        region_stats['Pre_%'] = (region_stats['Pre_Correct'] / region_stats['Pre_Total'] * 100).round(2)
        region_stats['Post_%'] = (region_stats['Post_Correct'] / region_stats['Post_Total'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=region_stats['Region'], 
            y=region_stats['Pre_%'], 
            mode='lines+markers+text',
            name='Pre-Session', 
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=[f"{val:.1f}%" for val in region_stats['Pre_%']],
            textposition='top center'
        ))
        fig.add_trace(go.Scatter(
            x=region_stats['Region'], 
            y=region_stats['Post_%'], 
            mode='lines+markers+text',
            name='Post-Session', 
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10),
            text=[f"{val:.1f}%" for val in region_stats['Post_%']],
            textposition='top center'
        ))
        fig.update_layout(
            title=f'{selected_category} - Performance by Region',
            xaxis_title='Region',
            yaxis_title='Correct Answer %',
            plot_bgcolor='#2b2b2b', 
            paper_bgcolor='#1e1e1e', 
            font=dict(color='white'), 
            height=450,
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040', tickangle=-45)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        display_region = region_stats[['Region', 'Pre_%', 'Post_%']].copy()
        display_region['Improvement'] = (display_region['Post_%'] - display_region['Pre_%']).round(2)
        display_region.columns = ['Region', 'Pre %', 'Post %', 'Improvement %']
        st.dataframe(display_region, hide_index=True, use_container_width=True)
    
    with tab2:
        st.markdown(f"#### {selected_category} - Performance by Grade")
        grade_stats = cat_data.groupby('Parent_Class').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        grade_stats.columns = ['Grade', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        grade_stats['Pre_%'] = (grade_stats['Pre_Correct'] / grade_stats['Pre_Total'] * 100).round(2)
        grade_stats['Post_%'] = (grade_stats['Post_Correct'] / grade_stats['Post_Total'] * 100).round(2)
        grade_stats = grade_stats.sort_values('Grade')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grade_stats['Grade'], 
            y=grade_stats['Pre_%'], 
            mode='lines+markers+text',
            name='Pre-Session', 
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=12),
            text=[f"{val:.1f}%" for val in grade_stats['Pre_%']],
            textposition='top center'
        ))
        fig.add_trace(go.Scatter(
            x=grade_stats['Grade'], 
            y=grade_stats['Post_%'], 
            mode='lines+markers+text',
            name='Post-Session', 
            line=dict(color='#e67e22', width=3),
            marker=dict(size=12),
            text=[f"{val:.1f}%" for val in grade_stats['Post_%']],
            textposition='top center'
        ))
        fig.update_layout(
            title=f'{selected_category} - Performance by Grade',
            xaxis_title='Grade',
            yaxis_title='Correct Answer %',
            plot_bgcolor='#2b2b2b', 
            paper_bgcolor='#1e1e1e', 
            font=dict(color='white'), 
            height=450,
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        display_grade = grade_stats[['Grade', 'Pre_%', 'Post_%']].copy()
        display_grade['Improvement'] = (display_grade['Post_%'] - display_grade['Pre_%']).round(2)
        display_grade.columns = ['Grade', 'Pre %', 'Post %', 'Improvement %']
        st.dataframe(display_grade, hide_index=True, use_container_width=True)
    
    with tab3:
        st.markdown(f"#### {selected_category} - Performance by Program Type")
        program_stats = cat_data.groupby('Program Type').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        program_stats.columns = ['Program Type', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        program_stats['Pre_%'] = (program_stats['Pre_Correct'] / program_stats['Pre_Total'] * 100).round(2)
        program_stats['Post_%'] = (program_stats['Post_Correct'] / program_stats['Post_Total'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'], 
            y=program_stats['Pre_%'], 
            name='Pre-Session', 
            marker_color='#3498db',
            text=[f"{val:.1f}%" for val in program_stats['Pre_%']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'], 
            y=program_stats['Post_%'], 
            name='Post-Session', 
            marker_color='#e74c3c',
            text=[f"{val:.1f}%" for val in program_stats['Post_%']],
            textposition='outside'
        ))
        fig.update_layout(
            title=f'{selected_category} - Performance by Program Type',
            xaxis_title='Program Type',
            yaxis_title='Correct Answer %',
            barmode='group', 
            plot_bgcolor='#2b2b2b', 
            paper_bgcolor='#1e1e1e', 
            font=dict(color='white'), 
            height=450,
            yaxis=dict(range=[0, 110], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        display_program = program_stats[['Program Type', 'Pre_%', 'Post_%', 'Pre_Total']].copy()
        display_program['Improvement'] = (display_program['Post_%'] - display_program['Pre_%']).round(2)
        display_program.columns = ['Program Type', 'Pre %', 'Post %', 'Questions', 'Improvement %']
        display_program = display_program[['Program Type', 'Pre %', 'Post %', 'Improvement %', 'Questions']]
        st.dataframe(display_program, hide_index=True, use_container_width=True)
    
    with tab5:
        st.markdown(f"#### {selected_category} - Performance by Difficulty Level")
        diff_stats = cat_data.groupby('Difficulty').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        diff_stats.columns = ['Difficulty', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        diff_stats['Pre_%'] = (diff_stats['Pre_Correct'] / diff_stats['Pre_Total'] * 100).round(2)
        diff_stats['Post_%'] = (diff_stats['Post_Correct'] / diff_stats['Post_Total'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=diff_stats['Difficulty'], 
            y=diff_stats['Pre_%'], 
            name='Pre-Session', 
            marker_color='#1abc9c',
            text=[f"{val:.1f}%" for val in diff_stats['Pre_%']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=diff_stats['Difficulty'], 
            y=diff_stats['Post_%'], 
            name='Post-Session', 
            marker_color='#e74c3c',
            text=[f"{val:.1f}%" for val in diff_stats['Post_%']],
            textposition='outside'
        ))
        fig.update_layout(
            title=f'{selected_category} - Performance by Difficulty',
            xaxis_title='Difficulty Level',
            yaxis_title='Correct Answer %',
            barmode='group', 
            plot_bgcolor='#2b2b2b', 
            paper_bgcolor='#1e1e1e', 
            font=dict(color='white'), 
            height=450,
            yaxis=dict(range=[0, 110], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        display_diff = diff_stats[['Difficulty', 'Pre_%', 'Post_%', 'Pre_Total']].copy()
        display_diff['Improvement'] = (display_diff['Post_%'] - display_diff['Pre_%']).round(2)
        display_diff.columns = ['Difficulty', 'Pre %', 'Post %', 'Questions', 'Improvement %']
        display_diff = display_diff[['Difficulty', 'Pre %', 'Post %', 'Improvement %', 'Questions']]
        st.dataframe(display_diff, hide_index=True, use_container_width=True)
    
    with tab4:
        st.markdown(f"#### {selected_category} - Performance by Topic (Top 15)")
        topic_stats = cat_data.groupby('Topic Name').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        topic_stats.columns = ['Topic', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        topic_stats['Pre_%'] = (topic_stats['Pre_Correct'] / topic_stats['Pre_Total'] * 100).round(2)
        topic_stats['Post_%'] = (topic_stats['Post_Correct'] / topic_stats['Post_Total'] * 100).round(2)
        topic_stats = topic_stats.nlargest(15, 'Pre_Total')  # Top 15 topics by volume
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=topic_stats['Topic'], 
            y=topic_stats['Pre_%'], 
            name='Pre-Session', 
            marker_color='#f39c12',
            text=[f"{val:.0f}%" for val in topic_stats['Pre_%']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=topic_stats['Topic'], 
            y=topic_stats['Post_%'], 
            name='Post-Session', 
            marker_color='#27ae60',
            text=[f"{val:.0f}%" for val in topic_stats['Post_%']],
            textposition='outside'
        ))
        fig.update_layout(
            title=f'{selected_category} - Performance by Topic (Top 15 by Volume)',
            xaxis_title='Topic',
            yaxis_title='Correct Answer %',
            barmode='group', 
            plot_bgcolor='#2b2b2b', 
            paper_bgcolor='#1e1e1e', 
            font=dict(color='white'), 
            height=450, 
            xaxis=dict(tickangle=-45, gridcolor='#404040'),
            yaxis=dict(range=[0, 110], gridcolor='#404040')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show full table
        all_topic_stats = cat_data.groupby('Topic Name').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        all_topic_stats.columns = ['Topic', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        all_topic_stats['Pre_%'] = (all_topic_stats['Pre_Correct'] / all_topic_stats['Pre_Total'] * 100).round(2)
        all_topic_stats['Post_%'] = (all_topic_stats['Post_Correct'] / all_topic_stats['Post_Total'] * 100).round(2)
        all_topic_stats['Improvement'] = (all_topic_stats['Post_%'] - all_topic_stats['Pre_%']).round(2)
        
        display_topic = all_topic_stats[['Topic', 'Pre_%', 'Post_%', 'Improvement', 'Pre_Total']].copy()
        display_topic.columns = ['Topic', 'Pre %', 'Post %', 'Improvement %', 'Questions']
        display_topic = display_topic.sort_values('Questions', ascending=False)
        
        st.dataframe(display_topic, hide_index=True, use_container_width=True, height=400)

# Download category analysis
st.markdown("---")
st.subheader("📥 Download Analysis Reports")

col1, col2 = st.columns(2)

with col1:
    category_csv = category_stats.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Overall Category Analysis",
        category_csv, 
        "category_analysis.csv", 
        "text/csv"
    )

with col2:
    if len(cat_data) > 0:
        # Create detailed report for selected category
        detailed_report = cat_data[['Student Id', 'Student Name', 'Class', 'Region', 'Subject', 
                                    'Topic Name', 'Question No', 'Difficulty',
                                    'Student Correct Answer Pre', 'Student Correct Answer Post']].copy()
        detailed_csv = detailed_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            f"📥 Download {selected_category} Detailed Data",
            detailed_csv,
            f"{selected_category.lower().replace(' ', '_')}_detailed.csv",
            "text/csv"
        )



