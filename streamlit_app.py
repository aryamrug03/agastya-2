import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Student Assessment Dashboard", layout="wide", page_icon="📊")

# ===== DATA CLEANING FUNCTIONS =====

def clean_and_process_data(df):
    """
    Clean and process student assessment data
    
    Parameters:
    df (pd.DataFrame): Raw dataframe from Excel
    
    Returns:
    pd.DataFrame: Cleaned and processed dataframe
    """
    
    initial_count = len(df)
    
    # ===== STEP 1: DATA CLEANING =====
    
    # Define pre and post question columns
    pre_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    post_questions = ['Q1_Post', 'Q2_Post', 'Q3_Post', 'Q4_Post', 'Q5_Post']
    
    # Condition 1: Remove rows where one set has values and the other is all NULL
    # If ANY pre question has values but ALL post questions are NULL
    has_any_pre = df[pre_questions].notna().any(axis=1)
    all_post_null = df[post_questions].isna().all(axis=1)
    remove_condition_1 = has_any_pre & all_post_null
    
    # If ALL pre questions are NULL but ANY post question has values
    all_pre_null = df[pre_questions].isna().all(axis=1)
    has_any_post = df[post_questions].notna().any(axis=1)
    remove_condition_2 = all_pre_null & has_any_post
    
    # Condition 3: Remove rows where BOTH pre and post are all NULL
    remove_condition_3 = all_pre_null & all_post_null
    
    # Remove rows that meet any of the conditions
    df = df[~(remove_condition_1 | remove_condition_2 | remove_condition_3)]
    
    cleaned_count = len(df)
    
    # ===== STEP 2: CALCULATE SCORES =====
    
    # Define answer columns
    pre_answers = ['Q1 Answer', 'Q2 Answer', 'Q3 Answer', 'Q4 Answer', 'Q5 Answer']
    post_answers = ['Q1_Answer_Post', 'Q2_Answer_Post', 'Q3_Answer_Post', 'Q4_Answer_Post', 'Q5_Answer_Post']
    
    # Calculate Pre-session scores
    df['Pre_Score'] = 0
    for q, ans in zip(pre_questions, pre_answers):
        df['Pre_Score'] += (df[q] == df[ans]).astype(int)
    
    # Calculate Post-session scores
    df['Post_Score'] = 0
    for q, ans in zip(post_questions, post_answers):
        df['Post_Score'] += (df[q] == df[ans]).astype(int)
    
    # ===== STEP 3: STANDARDIZE PROGRAM TYPES =====
    
    # Create a mapping for program types
    program_type_mapping = {
        'SC': 'PCMB',
        'SC2': 'PCMB',
        'SCB': 'PCMB',
        'SCC': 'PCMB',
        'SCM': 'PCMB',
        'SCP': 'PCMB',
        'E-LOB': 'ELOB',
        'DLC-2': 'DLC',
        'DLC2': 'DLC'
    }
    
    # Apply the mapping
    df['Program Type'] = df['Program Type'].replace(program_type_mapping)
    
    # ===== STEP 4: CREATE PARENT CLASS =====
    
    # Extract parent class from Class column (e.g., "6-A" -> "6", "7-B" -> "7")
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    
    # Filter for grades 6-10 only
    df = df[df['Parent_Class'].isin(['6', '7', '8', '9', '10'])]
    
    # ===== STEP 5: CALCULATE TEST FREQUENCY =====
    
    # Count how many times each student has taken tests
    df['Test_Count'] = df.groupby('Student Id')['Student Id'].transform('count')
    
    return df, initial_count, cleaned_count

# ===== MAIN APPLICATION =====

# Title and description
st.title("📊 Student Assessment Analysis Platform")
st.markdown("### Upload, Clean, and Analyze Student Performance Data")

# File uploader
uploaded_file = st.file_uploader("Upload Student Data Excel File", type=['xlsx', 'xls'])

if uploaded_file is not None:
    
    # Load and clean data
    with st.spinner("Loading and cleaning data..."):
        try:
            raw_df = pd.read_excel(uploaded_file)
            df, initial_count, cleaned_count = clean_and_process_data(raw_df)
            
            # Show cleaning summary
            st.success("✅ Data cleaned successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Records", initial_count)
            with col2:
                st.metric("Records Removed", initial_count - cleaned_count)
            with col3:
                st.metric("Final Records", cleaned_count)
            
            # Option to download cleaned data
            st.markdown("---")
            cleaned_excel = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Data (CSV)",
                data=cleaned_excel,
                file_name="cleaned_student_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Region filter
    all_regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", all_regions)
    
    # Program Type filter
    all_programs = ['All'] + sorted(df['Program Type'].unique().tolist())
    selected_program = st.sidebar.selectbox("Select Program Type", all_programs)
    
    # Parent Class filter
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
    
    # ===== KEY METRICS =====
    st.markdown("---")
    st.subheader(" Key Performance Metrics")
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
    
    # ===== TABS FOR DIFFERENT ANALYSES =====
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Region Analysis", "👤 Instructor Analysis", "📚 Grade Analysis", "📊 Program Type Analysis", "👥 Student Participation"])
    
    # ===== TAB 1: REGION ANALYSIS =====
    with tab1:
        st.header("Region-wise Performance Analysis")
        
        # Overall region analysis
        region_stats = filtered_df.groupby('Region').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        region_stats['Pre_Score_Pct'] = (region_stats['Pre_Score'] / 5) * 100
        region_stats['Post_Score_Pct'] = (region_stats['Post_Score'] / 5) * 100
        region_stats['Improvement'] = region_stats['Post_Score_Pct'] - region_stats['Pre_Score_Pct']
        region_stats = region_stats.sort_values('Region')
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session Average',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=12, color='#2ecc71')
        ))
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session Average',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=12, color='#e67e22')
        ))
        
        fig.update_layout(
            title='Region-wise Performance Comparison',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            hovermode='x unified',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Region by Program Type
        st.subheader("Region Analysis by Program Type")
        
        program_region_stats = filtered_df.groupby(['Region', 'Program Type']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        selected_program_type = st.selectbox("Select Program Type for Detailed View", 
                                             sorted(filtered_df['Program Type'].unique()))
        
        prog_data = program_region_stats[program_region_stats['Program Type'] == selected_program_type]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=prog_data['Region'],
            y=prog_data['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in prog_data['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.add_trace(go.Scatter(
            x=prog_data['Region'],
            y=prog_data['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in prog_data['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.update_layout(
            title=f'{selected_program_type} - Region-wise Performance',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Top performing and most improved regions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Top Scoring Regions (Post-Session)")
            top_scoring = region_stats.nlargest(5, 'Post_Score_Pct')[['Region', 'Post_Score_Pct']]
            top_scoring['Post_Score_Pct'] = top_scoring['Post_Score_Pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_scoring, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader(" Most Improved Regions (Adaptation)")
            most_improved = region_stats.nlargest(5, 'Improvement')[['Region', 'Improvement']]
            most_improved['Improvement'] = most_improved['Improvement'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(most_improved, hide_index=True, use_container_width=True)
    
    # ===== TAB 2: INSTRUCTOR ANALYSIS =====
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
        
        # Show top N instructors
        top_n = st.slider("Number of instructors to display", 5, 20, 10)
        top_instructors = instructor_stats.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in top_instructors['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in top_instructors['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Instructors by Post-Session Performance',
            xaxis_title='Instructor',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(tickangle=-45, gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Instructor rankings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Top Performing Instructors")
            top_perf = instructor_stats.nlargest(10, 'Post_Score_Pct')[['Instructor Name', 'Post_Score_Pct', 'Student Id']]
            top_perf.columns = ['Instructor', 'Post Score %', 'Students']
            top_perf['Post Score %'] = top_perf['Post Score %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_perf, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader(" Best Adaptation (Improvement)")
            best_adapt = instructor_stats.nlargest(10, 'Improvement')[['Instructor Name', 'Improvement', 'Student Id']]
            best_adapt.columns = ['Instructor', 'Improvement %', 'Students']
            best_adapt['Improvement %'] = best_adapt['Improvement %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(best_adapt, hide_index=True, use_container_width=True)
        
        # All Instructors Assessment Count
        st.markdown("---")
        st.subheader(" Complete Instructor List - Assessment Count")
        
        # Calculate number of assessments (Content Id) per instructor
        all_instructors = filtered_df.groupby('Instructor Name').agg({
            'Content Id': 'nunique',
            'Student Id': 'count',
            'Region': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]  # Most common region
        }).reset_index()
        
        all_instructors.columns = ['Instructor Name', 'Number of Assessments', 'Total Students', 'Primary Region']
        all_instructors = all_instructors.sort_values('Number of Assessments', ascending=False)
        
        # Add search functionality
        search_instructor = st.text_input("🔍 Search for an instructor", "")
        
        if search_instructor:
            filtered_instructors = all_instructors[
                all_instructors['Instructor Name'].str.contains(search_instructor, case=False, na=False)
            ]
        else:
            filtered_instructors = all_instructors
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Instructors", len(all_instructors))
        with col2:
            st.metric("Avg Assessments per Instructor", f"{all_instructors['Number of Assessments'].mean():.1f}")
        with col3:
            st.metric("Max Assessments by One Instructor", all_instructors['Number of Assessments'].max())
        
        # Display the full table
        st.dataframe(
            filtered_instructors,
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # Download option for instructor assessment data
        instructor_csv = all_instructors.to_csv(index=False)
        st.download_button(
            "📥 Download Complete Instructor Assessment List",
            instructor_csv,
            "instructor_assessments.csv",
            "text/csv"
        )
        
        # Instructors per Region
        st.markdown("---")
        st.subheader(" Number of Instructors per Region")
        
        instructors_per_region = filtered_df.groupby('Region')['Instructor Name'].nunique().reset_index()
        instructors_per_region.columns = ['Region', 'Number of Instructors']
        instructors_per_region = instructors_per_region.sort_values('Number of Instructors', ascending=False)
        
        # Create bar chart
        fig_inst_region = go.Figure()
        
        fig_inst_region.add_trace(go.Bar(
            x=instructors_per_region['Region'],
            y=instructors_per_region['Number of Instructors'],
            marker_color='#3498db',
            text=instructors_per_region['Number of Instructors'],
            textposition='outside',
            textfont=dict(size=14)
        ))
        
        fig_inst_region.update_layout(
            title='Number of Instructors by Region',
            xaxis_title='Region',
            yaxis_title='Number of Instructors',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig_inst_region, use_container_width=True)
        
        # Display table
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(instructors_per_region, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Unique Instructors", filtered_df['Instructor Name'].nunique())
            st.metric("Average per Region", f"{instructors_per_region['Number of Instructors'].mean():.1f}")
    
    # ===== TAB 3: GRADE ANALYSIS =====
    with tab3:
        st.header("Grade-wise Performance Analysis")
        
        grade_stats = filtered_df.groupby('Parent_Class').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        grade_stats['Pre_Score_Pct'] = (grade_stats['Pre_Score'] / 5) * 100
        grade_stats['Post_Score_Pct'] = (grade_stats['Post_Score'] / 5) * 100
        grade_stats['Improvement'] = grade_stats['Post_Score_Pct'] - grade_stats['Pre_Score_Pct']
        grade_stats = grade_stats.sort_values('Parent_Class')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#1abc9c', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        fig.update_layout(
            title='Grade-wise Performance Comparison',
            xaxis_title='Grade',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Grade statistics table
        st.subheader("Detailed Grade Statistics")
        display_stats = grade_stats.copy()
        display_stats.columns = ['Grade', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_stats = display_stats[['Grade', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        display_stats['Pre %'] = display_stats['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Post %'] = display_stats['Post %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Improvement %'] = display_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_stats, hide_index=True, use_container_width=True)
    
    # ===== TAB 4: PROGRAM TYPE ANALYSIS =====
    with tab4:
        st.header("Program Type Performance Analysis")
        
        program_stats = filtered_df.groupby('Program Type').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        program_stats['Pre_Score_Pct'] = (program_stats['Pre_Score'] / 5) * 100
        program_stats['Post_Score_Pct'] = (program_stats['Post_Score'] / 5) * 100
        program_stats['Improvement'] = program_stats['Post_Score_Pct'] - program_stats['Pre_Score_Pct']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Pre_Score_Pct'],
            name='Pre-Session',
            marker_color='#3498db',
            text=[f"{val:.0f}%" for val in program_stats['Pre_Score_Pct']],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Post_Score_Pct'],
            name='Post-Session',
            marker_color='#e74c3c',
            text=[f"{val:.0f}%" for val in program_stats['Post_Score_Pct']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Program Type Performance Comparison',
            xaxis_title='Program Type',
            yaxis_title='Average Score (%)',
            barmode='group',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 110], gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Program stats table
        st.subheader("Program Type Statistics")
        display_prog = program_stats.copy()
        display_prog.columns = ['Program', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_prog = display_prog[['Program', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        display_prog['Pre %'] = display_prog['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_prog['Post %'] = display_prog['Post %'].apply(lambda x: f"{x:.1f}%")
        display_prog['Improvement %'] = display_prog['Improvement %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_prog, hide_index=True, use_container_width=True)
    
    # ===== TAB 5: STUDENT PARTICIPATION =====
    with tab5:
        st.header("Student Participation Analysis")
        st.markdown("### Number of Unique Students Taking Assessments")
        
        # Students per Grade
        st.subheader(" Students per Grade/Parent Class")
        students_per_grade = filtered_df.groupby('Parent_Class')['Student Id'].nunique().reset_index()
        students_per_grade.columns = ['Grade', 'Number of Students']
        students_per_grade = students_per_grade.sort_values('Grade')
        
        fig_grade = go.Figure()
        fig_grade.add_trace(go.Bar(
            x=students_per_grade['Grade'],
            y=students_per_grade['Number of Students'],
            marker_color='#1abc9c',
            text=students_per_grade['Number of Students'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_grade.update_layout(
            title='Number of Students by Grade',
            xaxis_title='Grade',
            yaxis_title='Number of Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig_grade, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(students_per_grade, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Students Across All Grades", students_per_grade['Number of Students'].sum())
            st.metric("Average Students per Grade", f"{students_per_grade['Number of Students'].mean():.0f}")
        
        # Students per Region
        st.markdown("---")
        st.subheader(" Students per Region")
        students_per_region = filtered_df.groupby('Region')['Student Id'].nunique().reset_index()
        students_per_region.columns = ['Region', 'Number of Students']
        students_per_region = students_per_region.sort_values('Number of Students', ascending=False)
        
        fig_region = go.Figure()
        fig_region.add_trace(go.Bar(
            x=students_per_region['Region'],
            y=students_per_region['Number of Students'],
            marker_color='#e67e22',
            text=students_per_region['Number of Students'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_region.update_layout(
            title='Number of Students by Region',
            xaxis_title='Region',
            yaxis_title='Number of Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040', tickangle=-45)
        )
        
        st.plotly_chart(fig_region, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(students_per_region, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Regions", len(students_per_region))
            st.metric("Average Students per Region", f"{students_per_region['Number of Students'].mean():.0f}")
        
        # Students per Program Type
        st.markdown("---")
        st.subheader(" Students per Program Type")
        students_per_program = filtered_df.groupby('Program Type')['Student Id'].nunique().reset_index()
        students_per_program.columns = ['Program Type', 'Number of Students']
        students_per_program = students_per_program.sort_values('Number of Students', ascending=False)
        
        fig_program = go.Figure()
        fig_program.add_trace(go.Bar(
            x=students_per_program['Program Type'],
            y=students_per_program['Number of Students'],
            marker_color='#9b59b6',
            text=students_per_program['Number of Students'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_program.update_layout(
            title='Number of Students by Program Type',
            xaxis_title='Program Type',
            yaxis_title='Number of Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig_program, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(students_per_program, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Program Types", len(students_per_program))
            st.metric("Average Students per Program", f"{students_per_program['Number of Students'].mean():.0f}")
        
        # Combined breakdown: Region x Program Type
        st.markdown("---")
        st.subheader(" Students by Region and Program Type")
        
        students_region_program = filtered_df.groupby(['Region', 'Program Type'])['Student Id'].nunique().reset_index()
        students_region_program.columns = ['Region', 'Program Type', 'Number of Students']
        
        # Pivot table for better visualization
        pivot_table = students_region_program.pivot(index='Region', columns='Program Type', values='Number of Students').fillna(0).astype(int)
        
        st.dataframe(pivot_table, use_container_width=True)
        
        # Heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            text=pivot_table.values,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Students")
        ))
        
        fig_heatmap.update_layout(
            title='Student Distribution Heatmap: Region vs Program Type',
            xaxis_title='Program Type',
            yaxis_title='Region',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Combined breakdown: Grade x Region
        st.markdown("---")
        st.subheader(" Students by Grade and Region")
        
        students_grade_region = filtered_df.groupby(['Parent_Class', 'Region'])['Student Id'].nunique().reset_index()
        students_grade_region.columns = ['Grade', 'Region', 'Number of Students']
        
        # Pivot table
        pivot_grade_region = students_grade_region.pivot(index='Grade', columns='Region', values='Number of Students').fillna(0).astype(int)
        
        st.dataframe(pivot_grade_region, use_container_width=True)
        
        # Download all participation data
        st.markdown("---")
        st.subheader("📥 Download Participation Reports")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            grade_csv = students_per_grade.to_csv(index=False)
            st.download_button("Download Grade Data", grade_csv, "students_per_grade.csv", "text/csv")
        with col2:
            region_csv = students_per_region.to_csv(index=False)
            st.download_button("Download Region Data", region_csv, "students_per_region.csv", "text/csv")
        with col3:
            program_csv = students_per_program.to_csv(index=False)
            st.download_button("Download Program Data", program_csv, "students_per_program.csv", "text/csv")
    
    # ===== DOWNLOAD SECTION =====
    st.markdown("---")
    st.subheader("📥 Download Analysis Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region_csv = region_stats.to_csv(index=False)
        st.download_button("Download Region Analysis", region_csv, "region_analysis.csv", "text/csv")
    
    with col2:
        instructor_csv = instructor_stats.to_csv(index=False)
        st.download_button("Download Instructor Analysis", instructor_csv, "instructor_analysis.csv", "text/csv")
    
    with col3:
        grade_csv = grade_stats.to_csv(index=False)
        st.download_button("Download Grade Analysis", grade_csv, "grade_analysis.csv", "text/csv")

else:
    st.info("👆 Please upload your student data Excel file to begin")
    
    st.markdown("---")
    st.subheader("📋 Required Excel Columns")
    st.markdown("""
    Your Excel file must contain these columns:
    
    **Identification Columns:**
    - `Region` - Geographic region
    - `Student Id` - Unique student identifier
    - `Class` - Class with section (e.g., 6-A, 7-B)
    - `Instructor Name` - Name of instructor
    - `Program Type` - Program type code
    
    **Pre-Session (Questions & Answers):**
    - `Q1`, `Q2`, `Q3`, `Q4`, `Q5` - Student responses
    - `Q1 Answer`, `Q2 Answer`, `Q3 Answer`, `Q4 Answer`, `Q5 Answer` - Correct answers
    
    **Post-Session (Questions & Answers):**
    - `Q1_Post`, `Q2_Post`, `Q3_Post`, `Q4_Post`, `Q5_Post` - Student responses
    - `Q1_Answer_Post`, `Q2_Answer_Post`, `Q3_Answer_Post`, `Q4_Answer_Post`, `Q5_Answer_Post` - Correct answers
    
    ---
    
    **What happens when you upload:**
    1. ✅ Data is cleaned (incomplete records removed)
    2. ✅ Scores are calculated automatically
    3. ✅ Program types are standardized
    4. ✅ Interactive dashboard is generated
    5. ✅ Download options for all analyses
    """)