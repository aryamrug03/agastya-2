import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Student Assessment Dashboard", layout="wide", page_icon="📊")

# ===== DATA CLEANING FUNCTIONS FOR FIRST SHEET =====

def clean_and_process_data(df):
    """Clean and process student assessment data"""
    initial_count = len(df)
    
    # Define pre and post question columns
    pre_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    post_questions = ['Q1_Post', 'Q2_Post', 'Q3_Post', 'Q4_Post', 'Q5_Post']
    
    # Condition 1: Remove rows where one set has values and the other is all NULL
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
    
    # Create parent class
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    df = df[df['Parent_Class'].isin(['6', '7', '8', '9', '10'])]
    
    # Calculate test frequency
    df['Test_Count'] = df.groupby('Student Id')['Student Id'].transform('count')
    
    return df, initial_count, cleaned_count

# ===== DATA PROCESSING FOR CATEGORY ANALYSIS (SECOND SHEET) =====

def process_category_data(df):
    """Process category-wise analysis data"""
    
    # Extract parent class from Class column
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    
    # Filter for grades 6-10 only
    df = df[df['Parent_Class'].isin(['6', '7', '8', '9', '10'])]
    
    return df

def calculate_category_metrics(df, category=None, region=None, subject=None, topic=None, grade=None):
    """Calculate category-wise performance metrics with optional filters"""
    
    filtered_df = df.copy()
    
    # Apply filters
    if category and category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category]
    if region and region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == region]
    if subject and subject != 'All':
        filtered_df = filtered_df[filtered_df['Subject'] == subject]
    if topic and topic != 'All':
        filtered_df = filtered_df[filtered_df['Topic Name'] == topic]
    if grade and grade != 'All':
        filtered_df = filtered_df[filtered_df['Parent_Class'] == grade]
    
    if len(filtered_df) == 0:
        return None
    
    # Group by category and calculate metrics
    category_stats = filtered_df.groupby('Category').agg({
        'Student Correct Answer Pre': ['sum', 'count'],
        'Student Correct Answer Post': ['sum', 'count']
    }).reset_index()
    
    # Flatten column names
    category_stats.columns = ['Category', 'Pre_Correct', 'Pre_Total', 'Post_Correct', 'Post_Total']
    
    # Calculate percentages
    category_stats['Pre_Percentage'] = (category_stats['Pre_Correct'] / category_stats['Pre_Total']) * 100
    category_stats['Post_Percentage'] = (category_stats['Post_Correct'] / category_stats['Post_Total']) * 100
    category_stats['Improvement'] = category_stats['Post_Percentage'] - category_stats['Pre_Percentage']
    category_stats['Lagging_Percentage'] = 100 - category_stats['Post_Percentage']
    
    return category_stats

# ===== MAIN APPLICATION =====

st.title("📊 Student Assessment Analysis Platform")
st.markdown("### Upload, Clean, and Analyze Student Performance Data")

# File uploaders
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📄 First Dataset (Original Assessment Data)")
    uploaded_file_1 = st.file_uploader("Upload Assessment Data Excel File", type=['xlsx', 'xls'], key='file1')

with col2:
    st.markdown("#### 📄 Second Dataset (Category Analysis Data)")
    uploaded_file_2 = st.file_uploader("Upload Category Analysis Excel File", type=['xlsx', 'xls'], key='file2')

# ===== SMART FILE DETECTION =====

def detect_file_type(df):
    """Detect whether file is Type 1 (original) or Type 2 (category analysis)"""
    columns = df.columns.tolist()
    
    # Type 1 indicators: Has Q1, Q2, etc. and Q1_Post, Q2_Post, etc.
    type1_indicators = ['Q1', 'Q2', 'Q1_Post', 'Q2_Post', 'Q1 Answer', 'Q1_Answer_Post']
    
    # Type 2 indicators: Has Category, Student Correct Answer Pre/Post
    type2_indicators = ['Category', 'Student Correct Answer Pre', 'Student Correct Answer Post', 
                       'Question Number', 'Difficulty']
    
    type1_score = sum(1 for col in type1_indicators if col in columns)
    type2_score = sum(1 for col in type2_indicators if col in columns)
    
    if type1_score > type2_score:
        return 'type1'
    elif type2_score > type1_score:
        return 'type2'
    else:
        return 'unknown'

# Process uploaded files with smart detection
df1 = None
df2 = None

files_to_process = []
if uploaded_file_1 is not None:
    files_to_process.append(('file1', uploaded_file_1))
if uploaded_file_2 is not None:
    files_to_process.append(('file2', uploaded_file_2))

for file_label, file in files_to_process:
    with st.spinner(f"Loading and analyzing {file_label}..."):
        try:
            raw_df = pd.read_excel(file)
            file_type = detect_file_type(raw_df)
            
            if file_type == 'type1':
                df1, initial_count, cleaned_count = clean_and_process_data(raw_df)
                st.success(f"✅ {file_label} identified as **Original Assessment Data** and loaded successfully!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial Records", initial_count)
                with col2:
                    st.metric("Records Removed", initial_count - cleaned_count)
                with col3:
                    st.metric("Final Records", cleaned_count)
            
            elif file_type == 'type2':
                df2 = process_category_data(raw_df)
                st.success(f"✅ {file_label} identified as **Category Analysis Data** and loaded successfully!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", len(df2))
                with col2:
                    st.metric("Categories Found", df2['Category'].nunique())
            
            else:
                st.error(f"❌ {file_label}: Could not identify file type. Please check column names.")
        
        except Exception as e:
            st.error(f"Error processing {file_label}: {str(e)}")

# Show tabs based on what's uploaded
if df1 is not None or df2 is not None:
    
    # Create tab list dynamically
    tab_list = []
    if df1 is not None:
        tab_list.extend(["📍 Region Analysis", "👤 Instructor Analysis", "📚 Grade Analysis", "📊 Program Type Analysis"])
    if df2 is not None:
        tab_list.append("🎯 Category Analysis")
    
    tabs = st.tabs(tab_list)
    tab_index = 0
    
    # ===== ORIGINAL TABS (if df1 exists) =====
    if df1 is not None:
        df = df1  # For backward compatibility
        
        # Sidebar filters for first dataset
        st.sidebar.header("🔍 Filters (First Dataset)")
        all_regions = ['All'] + sorted(df['Region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Select Region", all_regions)
        
        all_programs = ['All'] + sorted(df['Program Type'].unique().tolist())
        selected_program = st.sidebar.selectbox("Select Program Type", all_programs)
        
        all_classes = ['All'] + sorted(df['Parent_Class'].unique().tolist())
        selected_class = st.sidebar.selectbox("Select Grade", all_classes)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_region != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == selected_region]
        if selected_program != 'All':
            filtered_df = filtered_df[filtered_df['Program Type'] == selected_program]
        if selected_class != 'All':
            filtered_df = filtered_df[filtered_df['Parent_Class'] == selected_class]
        
        # Key metrics
        st.markdown("---")
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
        
        # Region Analysis Tab
        with tabs[tab_index]:
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
                x=region_stats['Region'], y=region_stats['Pre_Score_Pct'],
                mode='lines+markers+text', name='Pre-Session',
                line=dict(color='#2ecc71', width=3), marker=dict(size=10),
                text=[f"{val:.0f}%" for val in region_stats['Pre_Score_Pct']],
                textposition='top center'
            ))
            fig.add_trace(go.Scatter(
                x=region_stats['Region'], y=region_stats['Post_Score_Pct'],
                mode='lines+markers+text', name='Post-Session',
                line=dict(color='#e67e22', width=3), marker=dict(size=10),
                text=[f"{val:.0f}%" for val in region_stats['Post_Score_Pct']],
                textposition='top center'
            ))
            fig.update_layout(
                title='Region-wise Performance Comparison',
                xaxis_title='Region', yaxis_title='Average Score (%)',
                height=500, plot_bgcolor='#2b2b2b', paper_bgcolor='#1e1e1e',
                font=dict(color='white'), yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        tab_index += 1
        
        # Instructor Analysis Tab
        with tabs[tab_index]:
            st.header("Instructor-wise Performance Analysis")
            
            instructor_stats = filtered_df.groupby('Instructor Name').agg({
                'Pre_Score': 'mean', 'Post_Score': 'mean', 'Student Id': 'count'
            }).reset_index()
            
            instructor_stats['Pre_Score_Pct'] = (instructor_stats['Pre_Score'] / 5) * 100
            instructor_stats['Post_Score_Pct'] = (instructor_stats['Post_Score'] / 5) * 100
            instructor_stats['Improvement'] = instructor_stats['Post_Score_Pct'] - instructor_stats['Pre_Score_Pct']
            instructor_stats = instructor_stats.sort_values('Post_Score_Pct', ascending=False)
            
            top_n = st.slider("Number of instructors to display", 5, 20, 10)
            top_instructors = instructor_stats.head(top_n)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=top_instructors['Instructor Name'], y=top_instructors['Pre_Score_Pct'],
                mode='lines+markers+text', name='Pre-Session',
                line=dict(color='#9b59b6', width=3), marker=dict(size=10),
                text=[f"{val:.0f}%" for val in top_instructors['Pre_Score_Pct']],
                textposition='top center'
            ))
            fig.add_trace(go.Scatter(
                x=top_instructors['Instructor Name'], y=top_instructors['Post_Score_Pct'],
                mode='lines+markers+text', name='Post-Session',
                line=dict(color='#f39c12', width=3), marker=dict(size=10),
                text=[f"{val:.0f}%" for val in top_instructors['Post_Score_Pct']],
                textposition='top center'
            ))
            fig.update_layout(
                title=f'Top {top_n} Instructors by Post-Session Performance',
                xaxis_title='Instructor', yaxis_title='Average Score (%)',
                height=500, plot_bgcolor='#2b2b2b', paper_bgcolor='#1e1e1e',
                font=dict(color='white'), xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        tab_index += 1
        
        # Grade Analysis Tab
        with tabs[tab_index]:
            st.header("Grade-wise Performance Analysis")
            
            grade_stats = filtered_df.groupby('Parent_Class').agg({
                'Pre_Score': 'mean', 'Post_Score': 'mean', 'Student Id': 'count'
            }).reset_index()
            
            grade_stats['Pre_Score_Pct'] = (grade_stats['Pre_Score'] / 5) * 100
            grade_stats['Post_Score_Pct'] = (grade_stats['Post_Score'] / 5) * 100
            grade_stats['Improvement'] = grade_stats['Post_Score_Pct'] - grade_stats['Pre_Score_Pct']
            grade_stats = grade_stats.sort_values('Parent_Class')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=grade_stats['Parent_Class'], y=grade_stats['Pre_Score_Pct'],
                mode='lines+markers+text', name='Pre-Session',
                line=dict(color='#1abc9c', width=3), marker=dict(size=12),
                text=[f"{val:.0f}%" for val in grade_stats['Pre_Score_Pct']],
                textposition='top center'
            ))
            fig.add_trace(go.Scatter(
                x=grade_stats['Parent_Class'], y=grade_stats['Post_Score_Pct'],
                mode='lines+markers+text', name='Post-Session',
                line=dict(color='#e67e22', width=3), marker=dict(size=12),
                text=[f"{val:.0f}%" for val in grade_stats['Post_Score_Pct']],
                textposition='top center'
            ))
            fig.update_layout(
                title='Grade-wise Performance Comparison',
                xaxis_title='Grade', yaxis_title='Average Score (%)',
                height=500, plot_bgcolor='#2b2b2b', paper_bgcolor='#1e1e1e',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        tab_index += 1
        
        # Program Type Analysis Tab
        with tabs[tab_index]:
            st.header("Program Type Performance Analysis")
            
            program_stats = filtered_df.groupby('Program Type').agg({
                'Pre_Score': 'mean', 'Post_Score': 'mean', 'Student Id': 'count'
            }).reset_index()
            
            program_stats['Pre_Score_Pct'] = (program_stats['Pre_Score'] / 5) * 100
            program_stats['Post_Score_Pct'] = (program_stats['Post_Score'] / 5) * 100
            program_stats['Improvement'] = program_stats['Post_Score_Pct'] - program_stats['Pre_Score_Pct']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=program_stats['Program Type'], y=program_stats['Pre_Score_Pct'],
                name='Pre-Session', marker_color='#3498db',
                text=[f"{val:.0f}%" for val in program_stats['Pre_Score_Pct']],
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                x=program_stats['Program Type'], y=program_stats['Post_Score_Pct'],
                name='Post-Session', marker_color='#e74c3c',
                text=[f"{val:.0f}%" for val in program_stats['Post_Score_Pct']],
                textposition='outside'
            ))
            fig.update_layout(
                title='Program Type Performance Comparison',
                xaxis_title='Program Type', yaxis_title='Average Score (%)',
                barmode='group', height=500, plot_bgcolor='#2b2b2b',
                paper_bgcolor='#1e1e1e', font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        tab_index += 1
    
    # ===== CATEGORY ANALYSIS TAB (if df2 exists) =====
    if df2 is not None and len(df2) > 0:
        with tabs[tab_index]:
            st.header("🎯 Category-wise Performance Analysis")
            st.markdown("### Analyze student performance across different question categories")
            
            # Filters for category analysis
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                try:
                    categories = ['All'] + sorted(df2['Category'].dropna().unique().tolist())
                except:
                    categories = ['All']
                selected_category = st.selectbox("📋 Category", categories, key='cat_filter')
            
            with col2:
                try:
                    regions = ['All'] + sorted(df2['Region'].dropna().unique().tolist())
                except:
                    regions = ['All']
                selected_region_cat = st.selectbox("📍 Region", regions, key='region_cat')
            
            with col3:
                try:
                    subjects = ['All'] + sorted(df2['Subject'].dropna().unique().tolist())
                except:
                    subjects = ['All']
                selected_subject = st.selectbox("📚 Subject", subjects, key='subject_cat')
            
            with col4:
                try:
                    topics = ['All'] + sorted(df2['Topic Name'].dropna().unique().tolist())
                except:
                    topics = ['All']
                selected_topic = st.selectbox("📖 Topic", topics, key='topic_cat')
            
            with col5:
                try:
                    grades = ['All'] + sorted(df2['Parent_Class'].dropna().unique().tolist())
                except:
                    grades = ['All']
                selected_grade_cat = st.selectbox("🎓 Grade", grades, key='grade_cat')
            
            # Calculate metrics
            category_stats = calculate_category_metrics(
                df2,
                category=selected_category if selected_category != 'All' else None,
                region=selected_region_cat if selected_region_cat != 'All' else None,
                subject=selected_subject if selected_subject != 'All' else None,
                topic=selected_topic if selected_topic != 'All' else None,
                grade=selected_grade_cat if selected_grade_cat != 'All' else None
            )
            
            if category_stats is not None and len(category_stats) > 0:
                # Key Metrics
                st.markdown("---")
                st.subheader("📊 Overall Category Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_pre_cat = category_stats['Pre_Percentage'].mean()
                    st.metric("Avg Pre-Session", f"{avg_pre_cat:.1f}%")
                with col2:
                    avg_post_cat = category_stats['Post_Percentage'].mean()
                    st.metric("Avg Post-Session", f"{avg_post_cat:.1f}%")
                with col3:
                    avg_improvement = category_stats['Improvement'].mean()
                    st.metric("Avg Improvement", f"{avg_improvement:.1f}%", 
                             delta=f"{avg_improvement:.1f}%")
                with col4:
                    avg_lagging = category_stats['Lagging_Percentage'].mean()
                    st.metric("Avg Lagging", f"{avg_lagging:.1f}%")
                
                # Main Performance Chart
                st.markdown("---")
                st.subheader("📈 Category Performance Comparison")
                
                fig1 = go.Figure()
                
                fig1.add_trace(go.Bar(
                    x=category_stats['Category'],
                    y=category_stats['Pre_Percentage'],
                    name='Pre-Session',
                    marker_color='#3498db',
                    text=[f"{val:.1f}%" for val in category_stats['Pre_Percentage']],
                    textposition='outside',
                    textfont=dict(size=12)
                ))
                
                fig1.add_trace(go.Bar(
                    x=category_stats['Category'],
                    y=category_stats['Post_Percentage'],
                    name='Post-Session',
                    marker_color='#2ecc71',
                    text=[f"{val:.1f}%" for val in category_stats['Post_Percentage']],
                    textposition='outside',
                    textfont=dict(size=12)
                ))
                
                fig1.update_layout(
                    title='Pre vs Post Session Performance by Category',
                    xaxis_title='Category',
                    yaxis_title='Percentage of Correct Answers (%)',
                    barmode='group',
                    height=500,
                    plot_bgcolor='#2b2b2b',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white', size=12),
                    yaxis=dict(range=[0, 110], gridcolor='#404040'),
                    xaxis=dict(gridcolor='#404040'),
                    legend=dict(x=0.01, y=0.99)
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Improvement Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Improvement by Category")
                    fig2 = go.Figure()
                    
                    colors = ['#2ecc71' if x > 0 else '#e74c3c' 
                             for x in category_stats['Improvement']]
                    
                    fig2.add_trace(go.Bar(
                        x=category_stats['Category'],
                        y=category_stats['Improvement'],
                        marker_color=colors,
                        text=[f"{val:+.1f}%" for val in category_stats['Improvement']],
                        textposition='outside',
                        textfont=dict(size=12),
                        showlegend=False
                    ))
                    
                    fig2.update_layout(
                        title='Improvement Percentage (Post - Pre)',
                        xaxis_title='Category',
                        yaxis_title='Improvement (%)',
                        height=400,
                        plot_bgcolor='#2b2b2b',
                        paper_bgcolor='#1e1e1e',
                        font=dict(color='white'),
                        yaxis=dict(gridcolor='#404040', zeroline=True, 
                                  zerolinecolor='white', zerolinewidth=2),
                        xaxis=dict(gridcolor='#404040')
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    st.subheader("⚠️ Lagging Percentage by Category")
                    fig3 = go.Figure()
                    
                    fig3.add_trace(go.Bar(
                        x=category_stats['Category'],
                        y=category_stats['Lagging_Percentage'],
                        marker_color='#e74c3c',
                        text=[f"{val:.1f}%" for val in category_stats['Lagging_Percentage']],
                        textposition='outside',
                        textfont=dict(size=12),
                        showlegend=False
                    ))
                    
                    fig3.update_layout(
                        title='Percentage Still Incorrect (Post-Session)',
                        xaxis_title='Category',
                        yaxis_title='Lagging (%)',
                        height=400,
                        plot_bgcolor='#2b2b2b',
                        paper_bgcolor='#1e1e1e',
                        font=dict(color='white'),
                        yaxis=dict(range=[0, 100], gridcolor='#404040'),
                        xaxis=dict(gridcolor='#404040')
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Detailed Statistics Table
                st.markdown("---")
                st.subheader("📋 Detailed Category Statistics")
                
                display_stats = category_stats.copy()
                display_stats['Pre_Percentage'] = display_stats['Pre_Percentage'].apply(lambda x: f"{x:.2f}%")
                display_stats['Post_Percentage'] = display_stats['Post_Percentage'].apply(lambda x: f"{x:.2f}%")
                display_stats['Improvement'] = display_stats['Improvement'].apply(lambda x: f"{x:+.2f}%")
                display_stats['Lagging_Percentage'] = display_stats['Lagging_Percentage'].apply(lambda x: f"{x:.2f}%")
                
                display_stats = display_stats[['Category', 'Pre_Percentage', 'Post_Percentage', 
                                               'Improvement', 'Lagging_Percentage', 'Pre_Total']]
                display_stats.columns = ['Category', 'Pre-Session (%)', 'Post-Session (%)', 
                                        'Improvement', 'Lagging (%)', 'Total Questions']
                
                st.dataframe(display_stats, hide_index=True, use_container_width=True)
                
                # Download button
                st.markdown("---")
                category_csv = category_stats.to_csv(index=False)
                st.download_button(
                    "📥 Download Category Analysis",
                    category_csv,
                    "category_analysis.csv",
                    "text/csv"
                )
                
            else:
                st.warning("⚠️ No data available for the selected filters. Please adjust your filter criteria.")

else:
    st.info("👆 Please upload at least one Excel file to begin analysis")
    
    st.markdown("---")
    st.markdown("### 📋 File Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### First Dataset (Original)")
        st.markdown("""
        Required columns:
        - Region, Student Id, Class
        - Instructor Name, Program Type
        - Q1-Q5 (Pre questions)
        - Q1_Post-Q5_Post (Post questions)
        - Answer columns for validation
        """)
    
    with col2:
        st.markdown("#### Second Dataset (Category Analysis)")
        st.markdown("""
        Required columns:
        - Region, Subject, Topic Name, Class
        - Category (Factual, Analytical, Application Base, Conceptual)
        - Student Correct Answer Pre
        - Student Correct Answer Post
        - Student Id, Question Number
        """)
