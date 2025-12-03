"""
TrendSpotter: The Automated Insight Engine
Main Pipeline Script - Ingest, Detect, Analyze, Report
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
INPUT_FILE = 'ad_traffic_data.csv'
OUTPUT_DIR = 'Output'
OUTPUT_PDF = os.path.join(OUTPUT_DIR, 'output.pdf')
CHART_IMAGE = os.path.join(OUTPUT_DIR, 'chart.png')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ingest_data(file_path):
    """
    Load CSV data using Pandas
    """
    print(f"[STEP 1/5] Ingesting data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def detect_anomalies(df):
    """
    Use Isolation Forest to detect anomalies in numerical columns
    """
    print("\n[STEP 2/5] Detecting anomalies using Isolation Forest...")
    
    # Select numerical features for anomaly detection
    features = ['clicks', 'impressions', 'spend']
    X = df[features].values
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.1,  # Expect ~10% anomalies
        random_state=42,
        n_estimators=100
    )
    
    # Predict anomalies (-1 = anomaly, 1 = normal)
    df['anomaly'] = iso_forest.fit_predict(X)
    df['anomaly_score'] = iso_forest.score_samples(X)
    
    # Filter anomalies
    anomalies = df[df['anomaly'] == -1].copy()
    print(f"✓ Detected {len(anomalies)} anomalies out of {len(df)} records")
    
    return df, anomalies


def analyze_anomalies(anomalies, df):
    """
    Create summary and get AI insights from Google Gemini
    """
    print("\n[STEP 3/5] Analyzing anomalies with Google Gemini AI...")
    
    if len(anomalies) == 0:
        return "No significant anomalies detected in the dataset."
    
    # Create anomaly summary
    summary_lines = []
    for idx, row in anomalies.head(10).iterrows():
        summary_lines.append(
            f"• Date: {row['date']}, Location: {row['location']}, "
            f"Campaign: {row['campaign_id']}, Clicks: {row['clicks']}, "
            f"Impressions: {row['impressions']}, Spend: ${row['spend']:.2f}"
        )
    
    anomaly_summary = "\n".join(summary_lines)
    
    # Configure Gemini
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your_gemini_api_key_here':
        print("⚠ Warning: GEMINI_API_KEY not set. Using mock analysis.")
        return generate_mock_analysis(anomalies)
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Craft the prompt
        prompt = f"""You are a Senior Data Analyst specializing in AdTech campaign performance.

Analyze these data anomalies from an advertising campaign:

{anomaly_summary}

Context:
- Total anomalies detected: {len(anomalies)}
- Time period: {df['date'].min()} to {df['date'].max()}
- Locations affected: {', '.join(anomalies['location'].unique())}

Task: Explain why these anomalies might be happening. Consider potential factors like:
- Seasonal trends or events
- Technical issues (ad server problems, targeting errors)
- External factors (weather, holidays, competition)
- Budget or bid adjustments

Provide your analysis in ONE professional paragraph (3-4 sentences). Be specific and data-driven."""

        response = model.generate_content(prompt)
        ai_insight = response.text.strip()
        print("✓ AI analysis complete")
        return ai_insight
        
    except Exception as e:
        print(f"⚠ Gemini API error: {e}. Using fallback analysis.")
        return generate_mock_analysis(anomalies)


def generate_mock_analysis(anomalies):
    """
    Fallback analysis when Gemini API is unavailable
    """
    locations = ', '.join(anomalies['location'].unique())
    avg_drop = anomalies['clicks'].mean()
    
    return (
        f"Analysis indicates significant performance deviations across {len(anomalies)} data points, "
        f"primarily affecting {locations}. The anomalies show an average click volume of {avg_drop:.0f}, "
        f"which deviates substantially from baseline performance. This pattern suggests potential "
        f"external factors such as regional events, technical disruptions, or seasonal fluctuations. "
        f"Further investigation into campaign settings and market conditions is recommended."
    )


def create_visualization(df):
    """
    Generate Week-over-Week performance chart using Plotly
    """
    print("\n[STEP 4/5] Creating visualizations...")
    
    # Aggregate data by date
    daily_data = df.groupby('date').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'spend': 'sum'
    }).reset_index()
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['clicks'],
        name='Clicks',
        mode='lines+markers',
        line=dict(color='#ff9500', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['impressions'] / 100,
        name='Impressions (÷100)',
        mode='lines+markers',
        line=dict(color='#333333', width=2),
        marker=dict(size=6)
    ))
    
    # Update layout
    fig.update_layout(
        title='Campaign Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    # Save as image
    fig.write_image(CHART_IMAGE, width=1000, height=400, scale=2)
    print(f"✓ Chart saved to {CHART_IMAGE}")
    
    return daily_data


def generate_pdf_report(df, anomalies, ai_insight, chart_path):
    """
    Generate professional PDF report using ReportLab
    """
    print("\n[STEP 5/5] Generating PDF report...")
    
    # Calculate key metrics
    total_clicks = df['clicks'].sum()
    total_impressions = df['impressions'].sum()
    total_spend = df['spend'].sum()
    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    
    # Create PDF document
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#000000'),
        spaceAfter=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#666666'),
        spaceAfter=5,
        alignment=TA_CENTER
    )
    
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#ff9500'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#000000'),
        spaceAfter=12,
        spaceBefore=15,
        fontName='Helvetica-Bold',
        borderColor=colors.HexColor('#ff9500'),
        borderWidth=0,
        borderPadding=5
    )
    
    # Title
    story.append(Paragraph("TrendSpotter Intelligence Report", title_style))
    story.append(Paragraph("Automated Insight Engine • Data-Driven Campaign Analysis", subtitle_style))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", date_style))
    
    # Metrics table
    metrics_data = [
        ['Total Clicks', 'Impressions', 'Total Spend', 'Avg CTR'],
        [f'{total_clicks:,}', f'{total_impressions:,}', f'${total_spend:,.2f}', f'{avg_ctr:.2f}%']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff9500')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#ff9500')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 14),
        ('TOPPADDING', (0, 1), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.white)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Performance chart
    story.append(Paragraph("📊 Performance Trend Analysis", heading_style))
    if os.path.exists(chart_path):
        img = Image(chart_path, width=6*inch, height=2.4*inch)
        story.append(img)
    story.append(Spacer(1, 0.2*inch))
    
    # AI Insights
    story.append(Paragraph("🔍 AI-Generated Insights", heading_style))
    insight_style = ParagraphStyle(
        'InsightBox',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#333333'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=10,
        spaceBefore=5
    )
    story.append(Paragraph(f"<b>Executive Summary:</b> {ai_insight}", insight_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Anomalies table
    story.append(Paragraph(f"⚠️ Detected Anomalies ({len(anomalies)} Total)", heading_style))
    
    anomaly_data = [['Date', 'Location', 'Campaign', 'Clicks', 'Impressions', 'Spend']]
    for idx, row in anomalies.head(15).iterrows():
        anomaly_data.append([
            row['date'],
            row['location'],
            row['campaign_id'],
            f"{row['clicks']:,}",
            f"{row['impressions']:,}",
            f"${row['spend']:.2f}"
        ])
    
    anomaly_table = Table(anomaly_data, colWidths=[0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.1*inch, 0.8*inch])
    anomaly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff9500')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff9f0')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    
    story.append(anomaly_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#999999'),
        alignment=TA_CENTER
    )
    story.append(Paragraph("TrendSpotter: The Automated Insight Engine", footer_style))
    story.append(Paragraph("Powered by Machine Learning (Isolation Forest) & Google Gemini AI", footer_style))
    
    # Build PDF
    doc.build(story)
    print(f"✓ Report successfully generated at {OUTPUT_PDF}")


def main():
    """
    Main pipeline execution
    """
    print("=" * 60)
    print("TrendSpotter: The Automated Insight Engine")
    print("=" * 60)
    
    try:
        # Step 1: Ingest data
        df = ingest_data(INPUT_FILE)
        
        # Step 2: Detect anomalies
        df, anomalies = detect_anomalies(df)
        print(f"\nAnomalies Detected: {len(anomalies)} records flagged for review")
        
        # Step 3: Analyze with AI
        ai_insight = analyze_anomalies(anomalies, df)
        
        # Step 4: Create visualizations
        daily_data = create_visualization(df)
        
        # Step 5: Generate PDF report
        generate_pdf_report(df, anomalies, ai_insight, CHART_IMAGE)
        
        print("\n" + "=" * 60)
        print("✓ Pipeline execution completed successfully")
        print(f"✓ Report Generated Successfully at {OUTPUT_PDF}")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"\n✗ Error: {INPUT_FILE} not found.")
        print("Run 'python generate_mock_data.py' to create sample data.")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
