import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Student Assessment Dashboard", layout="wide", page_icon="📊")

# ===== SHEET DETECTION FUNCTIONS =====

def detect_sheet_type(df):
    """
    Detect which type of sheet is uploaded based on columns
    Returns: 'category', 'assessment', 'both', or None
    """
    category_columns = ['Category', 'Student Correct Answer Pre', 'Student Correct Answer Post', 
                       'Question No', 'Difficulty']
    assessment_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q1 Answer', 'Q1_Post']
    
    has_category = all(col in df.columns for col in category_columns)
    has_assessment = all(col in df.columns for col in assessment_columns)
    
    if has_category and has_assessment:
        return 'both'
    elif has_category:
        return 'category'
    elif has_assessment:
        return 'assessment'
    else:
        return None

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
    
    # Standardize category names
    df['Category'] = df['Category'].str.strip().str.title()
    
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
        'Student Correct Answer Post': ['sum', 'count'],
        'Student Id': 'nunique'
    }).reset_index()
    
    category_stats.columns = ['Category', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total', 'Students']
    
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

# ===== ASSESSMENT DATA CLEANING FUNCTIONS (EXISTING CODE) =====

def clean_and_process_data(df):
    """
    Clean and process student assessment data
    """
    initial_count = len(df)
    
    # Define pre and post question columns
    pre_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    post_questions = ['Q1_Post', 'Q2_Post', 'Q3_Post', 'Q4_Post', 'Q5_Post']
    
    # Remove incomplete records
    has_any_pre = df[pre_questions].notna().any(axis=1)
    all_post_null = df[post_questions].isna().all(axis=1)
    remove_condition_1 = has_any_pre & all_post_null
    
    all_pre_null = df[pre_questions].isna().all(axis=1)
    has_any_post = df[post_questions].notna().any(axis=1)
    remove_condition_2 = all_pre_null & has_any_post
    
    remove_condition_3 = all_pre_null & all_post_null
    
    df = df[~(remove_condition_1 | remove_condition_2 | remove_condition_3)]
    cleaned_count = len(df)
    
    # Calculate scores
    pre_answers = ['Q1 Answer', 'Q2 Answer', 'Q3 Answer', 'Q4 Answer', 'Q5 Answer']
    post_answers = ['Q1_Answer_Post', 'Q2_Answer_Post', 'Q3_Answer_Post', 'Q4_Answer_Post', 'Q5_Answer_Post']
    
    df['Pre_Score'] = 0
    for q, ans in zip(pre_questions, pre_answers):
        df['Pre_Score'] += (df[q] == df[ans]).astype(int)
    
    df['Post_Score'] = 0
    for q, ans in zip(post_questions, post_answers):
        df['Post_Score'] += (df[q] == df[ans]).astype(int)
    
    # Standardize program types
    program_type_mapping = {
        'SC': 'PCMB', 'SC2': 'PCMB', 'SCB': 'PCMB', 'SCC': 'PCMB',
        'SCM': 'PCMB', 'SCP': 'PCMB', 'E-LOB': 'ELOB', 
        'DLC-2': 'DLC', 'DLC2': 'DLC'
    }
    df['Program Type'] = df['Program Type'].replace(program_type_mapping)
    
    # Extract parent class
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    df = df[df['Parent_Class'].isin(['6', '7', '8', '9', '10'])]
    
    # Calculate test frequency
    df['Test_Count'] = df.groupby('Student Id')['Student Id'].transform('count')
    
    return df, initial_count, cleaned_count

# ===== MAIN APPLICATION =====

st.title("📊 Student Assessment Analysis Platform")
st.markdown("### Upload, Clean, and Analyze Student Performance Data")

# File uploaders
st.markdown("#### Upload Data Files")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Category Analysis Data** (Question-level)")
    category_file = st.file_uploader("Upload Category Analysis Excel", type=['xlsx', 'xls'], key='category')

with col2:
    st.markdown("**Assessment Data** (Test-level)")
    assessment_file = st.file_uploader("Upload Assessment Excel", type=['xlsx', 'xls'], key='assessment')

# Determine what data we have
category_df = None
assessment_df = None
show_category_tabs = False
show_assessment_tabs = False

if category_file is not None:
    try:
        with st.spinner("Loading category data..."):
            category_df = pd.read_excel(category_file)
            
            # Debug: Show columns found
            st.info(f"Category file columns detected: {len(category_df.columns)} columns")
            
            sheet_type = detect_sheet_type(category_df)
            
            if sheet_type in ['category', 'both']:
                category_df = process_category_data(category_df)
                show_category_tabs = True
                st.success(f"✅ Category data loaded: {len(category_df)} records")
            elif sheet_type == 'assessment':
                st.warning("⚠️ This file appears to be Assessment data. Please upload it in the Assessment section.")
            else:
                st.error(f"❌ Category file missing required columns. Need: Category, Student Correct Answer Pre, Student Correct Answer Post, Question No, Difficulty")
                st.info("Available columns: " + ", ".join(category_df.columns.tolist()[:10]) + "...")
    except Exception as e:
        st.error(f"Error loading category file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if assessment_file is not None:
    try:
        with st.spinner("Loading assessment data..."):
            raw_df = pd.read_excel(assessment_file)
            
            # Debug: Show columns found
            st.info(f"Assessment file columns detected: {len(raw_df.columns)} columns")
            
            sheet_type = detect_sheet_type(raw_df)
            
            if sheet_type in ['assessment', 'both']:
                assessment_df, initial_count, cleaned_count = clean_and_process_data(raw_df)
                show_assessment_tabs = True
                
                st.success("✅ Assessment data loaded and cleaned!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial Records", initial_count)
                with col2:
                    st.metric("Records Removed", initial_count - cleaned_count)
                with col3:
                    st.metric("Final Records", cleaned_count)
            elif sheet_type == 'category':
                st.warning("⚠️ This file appears to be Category data. Please upload it in the Category section.")
            else:
                st.error(f"❌ Assessment file missing required columns. Need: Q1-Q5, Q1 Answer-Q5 Answer, Q1_Post-Q5_Post, Q1_Answer_Post-Q5_Answer_Post")
                st.info("Available columns: " + ", ".join(raw_df.columns.tolist()[:10]) + "...")
    except Exception as e:
        st.error(f"Error loading assessment file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Only proceed if we have data
if not show_category_tabs and not show_assessment_tabs:
    st.info("👆 Please upload at least one data file to begin analysis")
    st.markdown("---")
    st.subheader("📋 File Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Category Analysis File Must Include:**")
        st.markdown("""
        - Category
        - Student Correct Answer Pre
        - Student Correct Answer Post
        - Question No
        - Difficulty
        - Region, Program Type, Subject, Topic Name, Class
        """)
    
    with col2:
        st.markdown("**Assessment File Must Include:**")
        st.markdown("""
        - Q1, Q2, Q3, Q4, Q5
        - Q1 Answer, Q2 Answer, etc.
        - Q1_Post, Q2_Post, etc.
        - Q1_Answer_Post, Q2_Answer_Post, etc.
        - Region, Program Type, Class, Instructor Name
        """)
    st.stop()

# ===== CATEGORY ANALYSIS TAB =====

if show_category_tabs:
    st.markdown("---")
    st.header("📚 Category-wise Performance Analysis")
    
    # Sidebar filters for category analysis
    st.sidebar.header("🔍 Category Analysis Filters")
    
    filters = {}
    filters['region'] = st.sidebar.selectbox("Region", ['All'] + sorted(category_df['Region'].unique().tolist()))
    filters['program_type'] = st.sidebar.selectbox("Program Type", ['All'] + sorted(category_df['Program Type'].unique().tolist()))
    filters['subject'] = st.sidebar.selectbox("Subject", ['All'] + sorted(category_df['Subject'].unique().tolist()))
    filters['topic'] = st.sidebar.selectbox("Topic Name", ['All'] + sorted(category_df['Topic Name'].unique().tolist()))
    filters['class'] = st.sidebar.selectbox("Grade", ['All'] + sorted(category_df['Parent_Class'].unique().tolist()))
    filters['difficulty'] = st.sidebar.selectbox("Difficulty", ['All'] + sorted(category_df['Difficulty'].unique().tolist()))
    
    # Calculate metrics
    category_stats = calculate_category_metrics(category_df, filters)
    
    # Overall category comparison
    st.subheader("📊 Overall Category Performance")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Pre-Session',
        x=category_stats['Category'],
        y=category_stats['Pre_Percentage'],
        marker_color='#3498db',
        text=[f"{val:.1f}%" for val in category_stats['Pre_Percentage']],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Post-Session',
        x=category_stats['Category'],
        y=category_stats['Post_Percentage'],
        marker_color='#2ecc71',
        text=[f"{val:.1f}%" for val in category_stats['Post_Percentage']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Category-wise Correct Answer Percentage',
        xaxis_title='Category',
        yaxis_title='Percentage Correct (%)',
        barmode='group',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 110], gridcolor='#404040')
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
            textposition='outside'
        ))
        
        fig_imp.update_layout(
            title='Improvement (Post - Pre)',
            xaxis_title='Category',
            yaxis_title='Improvement (%)',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040')
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
            textposition='outside'
        ))
        
        fig_lag.update_layout(
            title='Students Who Regressed (Correct → Incorrect)',
            xaxis_title='Category',
            yaxis_title='Lagging (%)',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig_lag, use_container_width=True)
    
    # Detailed statistics table
    st.subheader("📋 Detailed Category Statistics")
    
    display_stats = category_stats.copy()
    display_stats = display_stats[['Category', 'Pre_Percentage', 'Post_Percentage', 'Improvement', 'Lagging_Percentage', 'Students']]
    display_stats.columns = ['Category', 'Pre %', 'Post %', 'Improvement %', 'Lagging %', 'Unique Students']
    
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["By Region", "By Grade", "By Difficulty", "By Topic"])
    
    with tab1:
        region_stats = cat_data.groupby('Region').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        region_stats.columns = ['Region', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        region_stats['Pre_%'] = (region_stats['Pre_Correct'] / region_stats['Pre_Total'] * 100).round(2)
        region_stats['Post_%'] = (region_stats['Post_Correct'] / region_stats['Post_Total'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=region_stats['Region'], y=region_stats['Pre_%'], 
                                mode='lines+markers', name='Pre', line=dict(color='#3498db', width=3)))
        fig.add_trace(go.Scatter(x=region_stats['Region'], y=region_stats['Post_%'], 
                                mode='lines+markers', name='Post', line=dict(color='#2ecc71', width=3)))
        fig.update_layout(title=f'{selected_category} - Performance by Region',
                         plot_bgcolor='#2b2b2b', paper_bgcolor='#1e1e1e', 
                         font=dict(color='white'), height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        grade_stats = cat_data.groupby('Parent_Class').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        grade_stats.columns = ['Grade', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        grade_stats['Pre_%'] = (grade_stats['Pre_Correct'] / grade_stats['Pre_Total'] * 100).round(2)
        grade_stats['Post_%'] = (grade_stats['Post_Correct'] / grade_stats['Post_Total'] * 100).round(2)
        grade_stats = grade_stats.sort_values('Grade')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grade_stats['Grade'], y=grade_stats['Pre_%'], 
                                mode='lines+markers', name='Pre', line=dict(color='#9b59b6', width=3)))
        fig.add_trace(go.Scatter(x=grade_stats['Grade'], y=grade_stats['Post_%'], 
                                mode='lines+markers', name='Post', line=dict(color='#e67e22', width=3)))
        fig.update_layout(title=f'{selected_category} - Performance by Grade',
                         plot_bgcolor='#2b2b2b', paper_bgcolor='#1e1e1e', 
                         font=dict(color='white'), height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        diff_stats = cat_data.groupby('Difficulty').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        diff_stats.columns = ['Difficulty', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        diff_stats['Pre_%'] = (diff_stats['Pre_Correct'] / diff_stats['Pre_Total'] * 100).round(2)
        diff_stats['Post_%'] = (diff_stats['Post_Correct'] / diff_stats['Post_Total'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=diff_stats['Difficulty'], y=diff_stats['Pre_%'], 
                            name='Pre', marker_color='#1abc9c'))
        fig.add_trace(go.Bar(x=diff_stats['Difficulty'], y=diff_stats['Post_%'], 
                            name='Post', marker_color='#e74c3c'))
        fig.update_layout(title=f'{selected_category} - Performance by Difficulty',
                         barmode='group', plot_bgcolor='#2b2b2b', paper_bgcolor='#1e1e1e', 
                         font=dict(color='white'), height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        topic_stats = cat_data.groupby('Topic Name').agg({
            'Student Correct Answer Pre': ['sum', 'count'],
            'Student Correct Answer Post': ['sum', 'count']
        }).reset_index()
        topic_stats.columns = ['Topic', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
        topic_stats['Pre_%'] = (topic_stats['Pre_Correct'] / topic_stats['Pre_Total'] * 100).round(2)
        topic_stats['Post_%'] = (topic_stats['Post_Correct'] / topic_stats['Post_Total'] * 100).round(2)
        topic_stats = topic_stats.nlargest(10, 'Pre_Total')  # Top 10 topics by volume
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=topic_stats['Topic'], y=topic_stats['Pre_%'], 
                            name='Pre', marker_color='#f39c12'))
        fig.add_trace(go.Bar(x=topic_stats['Topic'], y=topic_stats['Post_%'], 
                            name='Post', marker_color='#27ae60'))
        fig.update_layout(title=f'{selected_category} - Performance by Topic (Top 10)',
                         barmode='group', plot_bgcolor='#2b2b2b', paper_bgcolor='#1e1e1e', 
                         font=dict(color='white'), height=400, xaxis=dict(tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)
    
    # Download category analysis
    st.markdown("---")
    category_csv = category_stats.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Category Analysis", category_csv, 
                      "category_analysis.csv", "text/csv")

# ===== ASSESSMENT ANALYSIS TABS (EXISTING CODE) =====

if show_assessment_tabs:
    df = assessment_df
    
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("🔍 Assessment Filters")
    
    all_regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", all_regions, key='assess_region')
    
    all_programs = ['All'] + sorted(df['Program Type'].unique().tolist())
    selected_program = st.sidebar.selectbox("Select Program Type", all_programs, key='assess_program')
    
    all_classes = ['All'] + sorted(df['Parent_Class'].unique().tolist())
    selected_class = st.sidebar.selectbox("Select Grade", all_classes, key='assess_class')
    
    # Apply filters
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_program != 'All':
        filtered_df = filtered_df[filtered_df['Program Type'] == selected_program]
    if selected_class != 'All':
        filtered_df = filtered_df[filtered_df['Parent_Class'] == selected_class]
    
    # Key metrics
    st.subheader("📊 Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pre = (filtered_df['Pre_Score'].mean() / 5) * 100
        st.metric("Avg Pre-Session Score", f"{avg_pre:.1f}%")
    
    with col2:
        avg_post = (filtered_df['Post_Score'].mean() / 5) * 100
        st.metric("Avg Post-Session Score", f"{avg_post:.1f}%")
    
    with col3:
        improvement = avg_post - avg_pre
        st.metric("Overall Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    
    with col4:
        avg_tests = filtered_df['Test_Count'].mean()
        st.metric("Avg Tests per Student", f"{avg_tests:.1f}")
    
    # Tabs for different analyses
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Region Analysis", "👤 Instructor Analysis", 
                                             "📚 Grade Analysis", "📊 Program Type Analysis", 
                                             "👥 Student Participation"])
    
    # TAB 1: REGION ANALYSIS
    with tab1:
        st.header("Region-wise Performance Analysis")
        
        region_stats = filtered_df.groupby('Region').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        region_stats['Pre_Score_Pct'] = (region_stats['Pre_Score'] / 5) * 100
        region_stats['Post_Score_Pct'] = (region_stats['Post_Score'] / 5) * 100
        region_stats['Improvement'] = region_stats['Post_Score_Pct'] - region_stats['Pre_Score_Pct']
        region_stats = region_stats.sort_values('Region')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session Average',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session Average',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig.update_layout(
            title='Region-wise Performance Comparison',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Scoring Regions")
            top_scoring = region_stats.nlargest(5, 'Post_Score_Pct')[['Region', 'Post_Score_Pct']]
            top_scoring['Post_Score_Pct'] = top_scoring['Post_Score_Pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_scoring, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("📈 Most Improved Regions")
            most_improved = region_stats.nlargest(5, 'Improvement')[['Region', 'Improvement']]
            most_improved['Improvement'] = most_improved['Improvement'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(most_improved, hide_index=True, use_container_width=True)
    
    # TAB 2: INSTRUCTOR ANALYSIS
    with tab2:
        st.header("Instructor-wise Performance Analysis")
        
        instructor_stats = filtered_df.groupby('Instructor Name').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        instructor_stats['Pre_Score_Pct'] = (instructor_stats['Pre_Score'] / 5) * 100
        instructor_stats['Post_Score_Pct'] = (instructor_stats['Post_Score'] / 5) * 100
        instructor_stats['Improvement'] = instructor_stats['Post_Score_Pct'] - instructor_stats['Pre_Score_Pct']
        instructor_stats = instructor_stats.sort_values('Post_Score_Pct', ascending=False)
        
        top_n = st.slider("Number of instructors to display", 5, 20, 10)
        top_instructors = instructor_stats.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Pre_Score_Pct'],
            mode='lines+markers',
            name='Pre-Session',
            line=dict(color='#9b59b6', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Post_Score_Pct'],
            mode='lines+markers',
            name='Post-Session',
            line=dict(color='#f39c12', width=3)
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Instructors',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: GRADE ANALYSIS
    with tab3:
        st.header("Grade-wise Performance Analysis")
        
        grade_stats = filtered_df.groupby('Parent_Class').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        grade_stats['Pre_Score_Pct'] = (grade_stats['Pre_Score'] / 5) * 100
        grade_stats['Post_Score_Pct'] = (grade_stats['Post_Score'] / 5) * 100
        grade_stats = grade_stats.sort_values('Parent_Class')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Pre_Score_Pct'],
            mode='lines+markers',
            name='Pre-Session',
            line=dict(color='#1abc9c', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Post_Score_Pct'],
            mode='lines+markers',
            name='Post-Session',
            line=dict(color='#e67e22', width=3)
        ))
        
        fig.update_layout(
            title='Grade-wise Performance',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: PROGRAM TYPE
    with tab4:
        st.header("Program Type Performance")
        
        program_stats = filtered_df.groupby('Program Type').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        program_stats['Pre_Score_Pct'] = (program_stats['Pre_Score'] / 5) * 100
        program_stats['Post_Score_Pct'] = (program_stats['Post_Score'] / 5) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Pre_Score_Pct'],
            name='Pre-Session',
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Post_Score_Pct'],
            name='Post-Session',
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title='Program Type Performance',
            barmode='group',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: STUDENT PARTICIPATION
    with tab5:
        st.header("Student Participation Analysis")
        
        students_per_grade = filtered_df.groupby('Parent_Class')['Student Id'].nunique().reset_index()
        students_per_grade.columns = ['Grade', 'Number of Students']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=students_per_grade['Grade'],
            y=students_per_grade['Number of Students'],
            marker_color='#1abc9c'
        ))
        
        fig.update_layout(
            title='Students by Grade',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
