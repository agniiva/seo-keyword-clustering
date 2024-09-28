import streamlit as st
import pandas as pd
import requests
import json
import time
import sqlite3
import torch
import logging
import os
from contextlib import contextmanager
from transformers import pipeline
import numpy as np
from collections import defaultdict
import networkx as nx

# Configure logging
logging.basicConfig(
    filename='keyword_clustering_app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Check if GPU is available and assign device
device = 0 if torch.cuda.is_available() else -1

# Database functions with context manager

@contextmanager
def create_connection(db_file):
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        logging.info(f"Connected to database: {db_file}")
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database: {e}")
        st.error(f"Database connection error: {e}")
        st.stop()
    finally:
        if conn:
            conn.close()
            logging.info(f"Closed database connection: {db_file}")

def create_tables(conn):
    """Create tables for keywords and SERP results."""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                volume INTEGER,
                intent TEXT,
                cluster_id INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS serp_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword_id INTEGER,
                title TEXT,
                url TEXT,
                FOREIGN KEY (keyword_id) REFERENCES keywords (id)
            )
        ''')
        conn.commit()
        logging.info("Tables created or verified successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating tables: {e}")
        st.error(f"Error creating tables: {e}")

def insert_keywords(conn, df_keywords):
    """Insert keywords into the database."""
    try:
        cursor = conn.cursor()
        records = df_keywords.to_records(index=False)
        cursor.executemany('''
            INSERT INTO keywords (keyword, volume)
            VALUES (?, ?)
        ''', records)
        conn.commit()
        logging.info(f"Inserted {len(records)} keywords into the database.")
    except sqlite3.Error as e:
        logging.error(f"Error inserting keywords: {e}")
        st.error(f"Error inserting keywords: {e}")

def update_intents(conn, intents):
    """Batch update intents in the database."""
    try:
        cursor = conn.cursor()
        cursor.executemany('''
            UPDATE keywords SET intent = ?
            WHERE id = ?
        ''', intents)
        conn.commit()
        logging.info(f"Updated intents for {len(intents)} keywords.")
    except sqlite3.Error as e:
        logging.error(f"Error updating intents: {e}")
        st.error(f"Error updating intents: {e}")

def update_clusters(conn, clusters):
    """Batch update cluster IDs in the database."""
    try:
        cursor = conn.cursor()
        cursor.executemany('''
            UPDATE keywords SET cluster_id = ?
            WHERE id = ?
        ''', clusters)
        conn.commit()
        logging.info(f"Updated clusters for {len(clusters)} keywords.")
    except sqlite3.Error as e:
        logging.error(f"Error updating clusters: {e}")
        st.error(f"Error updating clusters: {e}")

def main():
    st.set_page_config(page_title="Keyword Clustering App", layout="wide")
    st.title("🔑 Keyword Clustering App")

    # List existing projects
    project_files = [f for f in os.listdir() if f.endswith('.db')]
    project_names = [os.path.splitext(f)[0] for f in project_files]

    # Project selection
    selected_project = st.sidebar.selectbox("Select or Create Project", ["Create New Project"] + project_names)
    if selected_project == "Create New Project":
        project_name = st.sidebar.text_input("Enter New Project Name")
        if project_name:
            st.session_state['project_name'] = project_name
        else:
            st.stop()
    else:
        st.session_state['project_name'] = selected_project

    if 'df_keywords' not in st.session_state:
        st.session_state['df_keywords'] = None
    if 'min_overlaps' not in st.session_state:
        st.session_state['min_overlaps'] = 3
    if 'intent_computed' not in st.session_state:
        st.session_state['intent_computed'] = False

    step = st.sidebar.selectbox("Navigate", ["Upload & Scrape", "Clustering", "Results"])

    if step == "Upload & Scrape":
        upload_and_scrape_page()
    elif step == "Clustering":
        clustering_page()
    elif step == "Results":
        results_page()

def upload_and_scrape_page():
    st.header("📤 Upload Data and Scrape SERPs")

    project_name = st.session_state['project_name']
    db_file = f"{project_name}.db"

    if os.path.exists(db_file):
        st.info(f"Project '{project_name}' already exists. Loading existing data.")
        if st.button("Proceed to Clustering"):
            st.session_state['intent_computed'] = False  # Reset intent computation
            st.experimental_rerun()
        return

    # Input for API key
    api_key = st.text_input("Enter Serper.dev API Key", type='password')
    if not api_key:
        st.warning("Please enter your Serper.dev API key.")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_file is not None:
        try:
            df_keywords = pd.read_csv(uploaded_file)
            if 'keyword' in df_keywords.columns and 'volume' in df_keywords.columns:
                st.success("File uploaded successfully!")
                st.session_state['df_keywords'] = df_keywords
                st.dataframe(df_keywords.head())
            else:
                st.error("CSV file must contain 'keyword' and 'volume' columns.")
                st.stop()
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            st.error(f"Error reading CSV file: {e}")
            st.stop()

    # Input for overlaps
    min_overlaps = st.number_input(
        "Minimum number of overlapping URLs for clustering",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get('min_overlaps', 3))
    )
    st.session_state['min_overlaps'] = min_overlaps

    # min_overlaps = st.number_input("Minimum number of overlapping URLs for clustering", min_value=1, max_value=10, value=3)
    # st.session_state['min_overlaps'] = min_overlaps

    # Button to save data and start scraping
    if st.button("Save Data and Start Scraping"):
        if 'df_keywords' not in st.session_state:
            st.error("Please upload a valid CSV file.")
            st.stop()
        
        api_key = api_key.strip()
        db_file = f"{project_name}.db"

        with st.spinner("Saving data and initializing database..."):
            try:
                with create_connection(db_file) as conn:
                    if conn:
                        create_tables(conn)
                        insert_keywords(conn, st.session_state['df_keywords'])
            except Exception as e:
                st.error(f"An error occurred while saving data: {str(e)}")
                st.stop()

        st.success("Data saved successfully!")

        # Start scraping
        scrape_serps(api_key, db_file)

def scrape_serps(api_key, db_file):
    st.header("🔍 Scrape SERPs")

    with st.spinner("Loading keywords to scrape..."):
        try:
            with create_connection(db_file) as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, keyword FROM keywords 
                        WHERE id NOT IN (SELECT DISTINCT keyword_id FROM serp_results)
                    """)
                    keywords = cursor.fetchall()
                    total_keywords = len(keywords)
        except Exception as e:
            st.error(f"Error loading keywords: {e}")
            st.stop()

    if total_keywords == 0:
        st.info("All keywords have been scraped.")
        return

    st.write(f"Total keywords to scrape: {total_keywords}")

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = 50
    num_batches = (total_keywords // batch_size) + (1 if total_keywords % batch_size != 0 else 0)

    with create_connection(db_file) as conn:
        if not conn:
            st.error("Database connection failed.")
            st.stop()
        cursor = conn.cursor()

        for batch_num in range(num_batches):
            batch_keywords = keywords[batch_num*batch_size : (batch_num+1)*batch_size]
            payload = [{"q": keyword} for _, keyword in batch_keywords]
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
                if response.status_code == 200:
                    results = response.json()
                    serp_records = []
                    for i, result in enumerate(results):
                        keyword_id = batch_keywords[i][0]
                        organic_results = result.get("organic", [])
                        for org in organic_results:
                            title = org.get('title', '')
                            link = org.get('link', '')
                            serp_records.append((keyword_id, title, link))
                    cursor.executemany('''
                        INSERT INTO serp_results (keyword_id, title, url)
                        VALUES (?, ?, ?)
                    ''', serp_records)
                    conn.commit()
                    logging.info(f"Batch {batch_num+1}/{num_batches} scraped and inserted successfully.")
                else:
                    logging.error(f"API request failed with status code {response.status_code}: {response.text}")
                    st.error(f"API request failed with status code {response.status_code}: {response.text}")
                    break
            except requests.exceptions.RequestException as e:
                logging.error(f"Exception during API request: {e}")
                st.error(f"Exception during API request: {e}")
                break

            # Update progress bar
            progress = (batch_num + 1) / num_batches
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Scraping Progress: {int(min(progress, 1.0) * 100)}%")

            if batch_num < num_batches - 1:
                st.info("Waiting for 30 seconds before next batch...")
                time.sleep(30)

    st.success("Scraping completed!")

def clustering_page():
    st.header("🗂️ Clustering")

    if 'project_name' not in st.session_state or not st.session_state['project_name']:
        st.error("Please go to the Upload & Scrape page first.")
        st.stop()

    project_name = st.session_state['project_name']
    db_file = f"{project_name}.db"

    # Add input field for minimum overlaps on the Clustering page
    min_overlaps = st.number_input(
        "Minimum number of overlapping URLs for clustering",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get('min_overlaps', 3))
    )
    st.session_state['min_overlaps'] = min_overlaps

    with st.spinner("Loading SERP data..."):
        try:
            with create_connection(db_file) as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT k.id, k.keyword, s.url
                        FROM keywords k
                        INNER JOIN serp_results s ON k.id = s.keyword_id
                    ''')
                    rows = cursor.fetchall()
        except Exception as e:
            st.error(f"Error loading SERP data: {e}")
            st.stop()

    if not rows:
        st.error("No SERP data found. Please scrape SERPs first.")
        st.stop()

    if st.button("Start Clustering"):
        # Build a mapping from keyword_id to set of URLs
        keyword_urls = defaultdict(set)
        for row in rows:
            keyword_id = row[0]
            url = row[2]
            keyword_urls[keyword_id].add(url)

        keyword_ids = list(keyword_urls.keys())
        num_keywords = len(keyword_ids)

        # Use the updated min_overlaps from st.session_state
        min_overlaps = st.session_state['min_overlaps']

        progress_bar = st.progress(0)
        status_text = st.empty()

        G = nx.Graph()
        G.add_nodes_from(keyword_ids)

        with st.spinner("Clustering in progress..."):
            try:
                for i in range(num_keywords):
                    keyword_id_i = keyword_ids[i]
                    urls_i = keyword_urls[keyword_id_i]
                    for j in range(i+1, num_keywords):
                        keyword_id_j = keyword_ids[j]
                        urls_j = keyword_urls[keyword_id_j]
                        overlap = len(urls_i.intersection(urls_j))
                        if overlap >= min_overlaps:
                            G.add_edge(keyword_id_i, keyword_id_j)
                    # Update progress bar
                    progress = (i + 1) / num_keywords
                    progress_bar.progress(progress)
                    status_text.text(f"Clustering Progress: {int(progress * 100)}%")
            except Exception as e:
                logging.error(f"Error during clustering: {e}")
                st.error(f"Error during clustering: {e}")
                st.stop()

        # Find connected components
        clusters = list(nx.connected_components(G))
        logging.info(f"Found {len(clusters)} clusters.")

        # Assign cluster IDs
        cluster_assignments = []
        for cluster_id, cluster in enumerate(clusters):
            for keyword_id in cluster:
                cluster_assignments.append((cluster_id, keyword_id))

        # Update cluster IDs in database
        with create_connection(db_file) as conn:
            if conn:
                update_clusters(conn, cluster_assignments)

        st.success("Clustering completed!")


# def clustering_page():
#     st.header("🗂️ Clustering")

#     if 'project_name' not in st.session_state or not st.session_state['project_name']:
#         st.error("Please go to the Upload & Scrape page first.")
#         st.stop()

#     project_name = st.session_state['project_name']
#     db_file = f"{project_name}.db"

#     with st.spinner("Loading SERP data..."):
#         try:
#             with create_connection(db_file) as conn:
#                 if conn:
#                     cursor = conn.cursor()
#                     cursor.execute('''
#                         SELECT k.id, k.keyword, s.url
#                         FROM keywords k
#                         INNER JOIN serp_results s ON k.id = s.keyword_id
#                     ''')
#                     rows = cursor.fetchall()
#         except Exception as e:
#             st.error(f"Error loading SERP data: {e}")
#             st.stop()

#     if not rows:
#         st.error("No SERP data found. Please scrape SERPs first.")
#         st.stop()

#     # Build a mapping from keyword_id to set of URLs
#     keyword_urls = defaultdict(set)
#     for row in rows:
#         keyword_id = row[0]
#         url = row[2]
#         keyword_urls[keyword_id].add(url)

#     keyword_ids = list(keyword_urls.keys())
#     num_keywords = len(keyword_ids)

#     if st.button("Start Clustering"):
#         progress_bar = st.progress(0)
#         status_text = st.empty()

#         G = nx.Graph()
#         G.add_nodes_from(keyword_ids)

#         min_overlaps = st.session_state.get('min_overlaps', 3)

#         with st.spinner("Clustering in progress..."):
#             try:
#                 for i in range(num_keywords):
#                     keyword_id_i = keyword_ids[i]
#                     urls_i = keyword_urls[keyword_id_i]
#                     for j in range(i+1, num_keywords):
#                         keyword_id_j = keyword_ids[j]
#                         urls_j = keyword_urls[keyword_id_j]
#                         overlap = len(urls_i.intersection(urls_j))
#                         if overlap >= min_overlaps:
#                             G.add_edge(keyword_id_i, keyword_id_j)
#                     # Update progress bar
#                     progress = (i + 1) / num_keywords
#                     progress_bar.progress(progress)
#                     status_text.text(f"Clustering Progress: {int(progress * 100)}%")
#             except Exception as e:
#                 logging.error(f"Error during clustering: {e}")
#                 st.error(f"Error during clustering: {e}")
#                 st.stop()

#         # Find connected components
#         clusters = list(nx.connected_components(G))
#         logging.info(f"Found {len(clusters)} clusters.")

#         # Assign cluster IDs
#         cluster_assignments = []
#         for cluster_id, cluster in enumerate(clusters):
#             for keyword_id in cluster:
#                 cluster_assignments.append((cluster_id, keyword_id))

#         # Update cluster IDs in database
#         with create_connection(db_file) as conn:
#             if conn:
#                 update_clusters(conn, cluster_assignments)

#         st.success("Clustering completed!")

        # Provide download option
        with create_connection(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT k.keyword, k.volume, k.cluster_id
                FROM keywords k
            ''')
            cluster_rows = cursor.fetchall()
            cluster_df = pd.DataFrame(cluster_rows, columns=['keyword', 'volume', 'cluster_id'])

        if st.button("Download Clusters as CSV"):
            csv_data = cluster_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Clusters CSV", data=csv_data, file_name='clusters.csv', mime='text/csv')

def results_page():
    st.header("📊 Results")

    if 'project_name' not in st.session_state or not st.session_state['project_name']:
        st.error("Please go to the Upload & Scrape page first.")
        st.stop()

    project_name = st.session_state['project_name']
    db_file = f"{project_name}.db"

    # Intent Classification
    if not st.session_state['intent_computed']:
        if st.button("Start Intent Classification"):
            with st.spinner("Classifying intents..."):
                try:
                    with create_connection(db_file) as conn:
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                SELECT keywords.id AS keyword_id, serp_results.title
                                FROM keywords
                                LEFT JOIN serp_results ON keywords.id = serp_results.keyword_id
                                GROUP BY keywords.id
                            ''')
                            rows = cursor.fetchall()
                    if not rows:
                        st.error("No data available for intent classification.")
                        st.stop()

                    # Initialize classifier with the specified model
                    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
                    labels = ["informational", "commercial", "navigational"]

                    intents = []
                    total = len(rows)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, row in enumerate(rows):
                        keyword_id, title = row
                        text = title if title else "N/A"
                        try:
                            result = classifier(text, candidate_labels=labels, truncation=True)
                            intent = result['labels'][0] if 'labels' in result and result['labels'] else 'unknown'
                        except Exception as e:
                            logging.error(f"Error classifying intent for keyword_id {keyword_id}: {e}")
                            intent = 'unknown'
                        intents.append((intent, keyword_id))

                        # Update progress
                        progress = (i + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Intent Classification Progress: {int(progress * 100)}%")

                    # Batch update intents
                    with create_connection(db_file) as conn:
                        if conn:
                            update_intents(conn, intents)

                    st.session_state['intent_computed'] = True
                    st.success("Intent classification completed!")
                except Exception as e:
                    logging.error(f"Error during intent classification: {e}")
                    st.error(f"Error during intent classification: {e}")
    else:
        st.info("Intent classification already completed.")

    # Load data from database
    with st.spinner("Loading results..."):
        try:
            with create_connection(db_file) as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT k.id AS keyword_id, k.keyword, k.volume, k.cluster_id, k.intent
                        FROM keywords k
                    ''')
                    rows = cursor.fetchall()
        except Exception as e:
            st.error(f"Error loading results: {e}")
            st.stop()

    if not rows:
        st.error("No data found. Please upload data and scrape SERPs first.")
        st.stop()

    df = pd.DataFrame(rows, columns=['keyword_id', 'keyword', 'volume', 'cluster_id', 'intent'])

    # Handle NaN values in 'volume'
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    # Group by clusters
    clusters = df.groupby('cluster_id')
    cluster_data = []

    for cluster_id, group in clusters:
        group = group.reset_index(drop=True)
        num_keywords = len(group)
        if num_keywords < 2:
            continue  # Skip clusters with less than 2 keywords

        cluster_volume = group['volume'].sum()
        dominant_intent = group['intent'].mode()[0] if not group['intent'].mode().empty else 'unknown'
        group = group.sort_values('volume', ascending=False)
        cluster_name = group.iloc[0]['keyword']
        cluster_keywords = group[['keyword', 'volume', 'intent']].to_dict('records')

        cluster_data.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'cluster_volume': cluster_volume,
            'dominant_intent': dominant_intent,
            'num_keywords': num_keywords,
            'keywords': cluster_keywords
        })

    # Sort clusters from biggest to smallest based on number of keywords
    cluster_data.sort(key=lambda x: x['num_keywords'], reverse=True)

    # Display clusters
    for cluster in cluster_data:
        with st.expander(f"🗂️ Cluster {cluster['cluster_id']} - {cluster['cluster_name']} ({cluster['num_keywords']} keywords)"):
            st.write(f"**Cluster Volume:** {cluster['cluster_volume']}")
            st.write(f"**Dominant Intent:** {cluster['dominant_intent']}")
            df_cluster = pd.DataFrame(cluster['keywords'])
            st.table(df_cluster)

    # Download button
    if st.button("📥 Download CSV"):
        try:
            output_df = pd.concat([pd.DataFrame({
                'cluster_id': cluster['cluster_id'],
                'cluster_name': cluster['cluster_name'],
                'cluster_volume': cluster['cluster_volume'],
                'dominant_intent': cluster['dominant_intent'],
                'keyword': [kw['keyword'] for kw in cluster['keywords']],
                'volume': [kw['volume'] for kw in cluster['keywords']],
                'intent': [kw['intent'] for kw in cluster['keywords']]
            }) for cluster in cluster_data], ignore_index=True)

            csv = output_df[['keyword', 'volume', 'cluster_id', 'cluster_name', 'intent', 'dominant_intent', 'cluster_volume']].to_csv(index=False).encode('utf-8')
            st.download_button(label="Download data as CSV", data=csv, file_name='clusters_with_intents.csv', mime='text/csv')
        except Exception as e:
            logging.error(f"Error preparing CSV for download: {e}")
            st.error(f"Error preparing CSV for download: {e}")


# def results_page():
#     st.header("📊 Results")

#     if 'project_name' not in st.session_state or not st.session_state['project_name']:
#         st.error("Please go to the Upload & Scrape page first.")
#         st.stop()

#     project_name = st.session_state['project_name']
#     db_file = f"{project_name}.db"

#     # Intent Classification
#     if not st.session_state['intent_computed']:
#         if st.button("Start Intent Classification"):
#             with st.spinner("Classifying intents..."):
#                 try:
#                     with create_connection(db_file) as conn:
#                         if conn:
#                             cursor = conn.cursor()
#                             cursor.execute('''
#                                 SELECT keywords.id AS keyword_id, serp_results.title
#                                 FROM keywords
#                                 LEFT JOIN serp_results ON keywords.id = serp_results.keyword_id
#                                 GROUP BY keywords.id
#                             ''')
#                             rows = cursor.fetchall()
#                     if not rows:
#                         st.error("No data available for intent classification.")
#                         st.stop()

#                     # Initialize classifier with the specified model
#                     classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
#                     labels = ["informational", "commercial", "navigational"]

#                     intents = []
#                     total = len(rows)
#                     progress_bar = st.progress(0)
#                     status_text = st.empty()

#                     for i, row in enumerate(rows):
#                         keyword_id, title = row
#                         text = title if title else "N/A"
#                         try:
#                             result = classifier(text, candidate_labels=labels, truncation=True)
#                             intent = result['labels'][0] if 'labels' in result and result['labels'] else 'unknown'
#                         except Exception as e:
#                             logging.error(f"Error classifying intent for keyword_id {keyword_id}: {e}")
#                             intent = 'unknown'
#                         intents.append((intent, keyword_id))

#                         # Update progress
#                         progress = (i + 1) / total
#                         progress_bar.progress(progress)
#                         status_text.text(f"Intent Classification Progress: {int(progress * 100)}%")

#                     # Batch update intents
#                     with create_connection(db_file) as conn:
#                         if conn:
#                             update_intents(conn, intents)

#                     st.session_state['intent_computed'] = True
#                     st.success("Intent classification completed!")
#                 except Exception as e:
#                     logging.error(f"Error during intent classification: {e}")
#                     st.error(f"Error during intent classification: {e}")
#     else:
#         st.info("Intent classification already completed.")

#     # Load data from database
#     with st.spinner("Loading results..."):
#         try:
#             with create_connection(db_file) as conn:
#                 if conn:
#                     cursor = conn.cursor()
#                     cursor.execute('''
#                         SELECT k.id AS keyword_id, k.keyword, k.volume, k.cluster_id, k.intent
#                         FROM keywords k
#                     ''')
#                     rows = cursor.fetchall()
#         except Exception as e:
#             st.error(f"Error loading results: {e}")
#             st.stop()

#     if not rows:
#         st.error("No data found. Please upload data and scrape SERPs first.")
#         st.stop()

#     df = pd.DataFrame(rows, columns=['keyword_id', 'keyword', 'volume', 'cluster_id', 'intent'])

#     # Handle NaN values in 'volume'
#     df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

#     # Group by clusters
#     clusters = df.groupby('cluster_id')
#     cluster_data = []
#     for cluster_id, group in clusters:
#         group = group.reset_index(drop=True)
#         if group.empty:
#             cluster_volume = 0
#             dominant_intent = 'unknown'
#             cluster_name = 'No Keywords'
#             cluster_keywords = []
#         else:
#             cluster_volume = group['volume'].sum()
#             dominant_intent = group['intent'].mode()[0] if not group['intent'].mode().empty else 'unknown'
#             group = group.sort_values('volume', ascending=False)
#             cluster_name = group.iloc[0]['keyword']
#             cluster_keywords = group[['keyword', 'volume', 'intent']].to_dict('records')
#         cluster_data.append({
#             'cluster_id': cluster_id,
#             'cluster_name': cluster_name,
#             'cluster_volume': cluster_volume,
#             'dominant_intent': dominant_intent,
#             'keywords': cluster_keywords
#         })

#     # Display clusters
#     for cluster in cluster_data:
#         with st.expander(f"🗂️ Cluster {cluster['cluster_id']} - {cluster['cluster_name']}"):
#             st.write(f"**Cluster Volume:** {cluster['cluster_volume']}")
#             st.write(f"**Dominant Intent:** {cluster['dominant_intent']}")
#             df_cluster = pd.DataFrame(cluster['keywords'])
#             st.table(df_cluster)

#     # Download button
#     if st.button("📥 Download CSV"):
#         try:
#             output_df = df.copy()
#             cluster_mapping = {c['cluster_id']: c['cluster_name'] for c in cluster_data}
#             intent_mapping = {c['cluster_id']: c['dominant_intent'] for c in cluster_data}
#             output_df['cluster_name'] = output_df['cluster_id'].map(cluster_mapping)
#             output_df['dominant_intent'] = output_df['cluster_id'].map(intent_mapping)
#             csv = output_df[['keyword', 'volume', 'cluster_id', 'cluster_name', 'intent', 'dominant_intent']].to_csv(index=False).encode('utf-8')
#             st.download_button(label="Download data as CSV", data=csv, file_name='clusters_with_intents.csv', mime='text/csv')
#         except Exception as e:
#             logging.error(f"Error preparing CSV for download: {e}")
#             st.error(f"Error preparing CSV for download: {e}")

if __name__ == "__main__":
    main()
