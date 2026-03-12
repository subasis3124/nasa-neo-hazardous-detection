import os
import random
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from alerts.alert_system.alert_queries import (
    WEEKLY_RISKY_ASTEROIDS_QUERY,
    SUMMARY_OF_WEEKLY_PREDICTION_QUERY,
    RECIPIENT_DATA_QUERY,
    SENDER_EMAIL
)

load_dotenv()


class AlertEmailService:

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.email_password = os.getenv("ALERTING_EMAIL_PASSWORD")
        self.sender_email = SENDER_EMAIL

        self.engine = create_engine(self.db_url)

    def fetch_data(self):
        """Load all required data from database"""
        risky_asteroids = pd.read_sql_query(WEEKLY_RISKY_ASTEROIDS_QUERY, self.engine)
        summary = pd.read_sql_query(SUMMARY_OF_WEEKLY_PREDICTION_QUERY, self.engine)
        recipients = pd.read_sql_query(RECIPIENT_DATA_QUERY, self.engine)

        return risky_asteroids, summary, recipients

    def load_template(self):
        """Load HTML email template"""
        template_path = Path("alerts/templates/success_email_template.html")
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    def render_template(self, html_template, risky_asteroids, summary):
        """Replace placeholders in template"""

        html_template = html_template.replace(
            "{{run_date}}", datetime.today().strftime("%Y-%m-%d")
        )

        html_template = html_template.replace(
            "{{total_asteroids}}", str(summary["total_asteroids"].iloc[0])
        )

        html_template = html_template.replace(
            "{{hazardous_count}}", str(summary["hazardous"].iloc[0])
        )

        html_template = html_template.replace(
            "{{safe_count}}", str(summary["non_hazardous"].iloc[0])
        )

        html_template = html_template.replace(
            "{{run_id}}", str(random.randint(10000, 100000))
        )

        html_template = html_template.replace(
            "{{dashboard_link}}",
            "https://subrat1920.grafana.net/public-dashboards/888774ac60ce422fb3f0abb526512867",
        )

        html_template = html_template.replace(
            "{{github_link}}",
            "https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection.git",
        )

        html_template = html_template.replace(
            "{{log_link}}",
            "https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions",
        )

        asteroid_names = risky_asteroids["name"].tolist()

        for i in range(1, 11):
            if i <= len(asteroid_names):
                html_template = html_template.replace(
                    f"{{{{asteroid_{i}}}}}", asteroid_names[i - 1]
                )
            else:
                html_template = html_template.replace(
                    f"{{{{asteroid_{i}}}}}", "—"
                )

        return html_template

    def send_emails(self, recipients, html_content):
        """Send email to all recipients"""

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(self.sender_email, self.email_password)

        for receiver in recipients:
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = receiver
            msg["Subject"] = "Nasa Near Earth Hazard Prediction Report - SUCCESS"

            msg.attach(MIMEText(html_content, "html"))

            server.sendmail(self.sender_email, receiver, msg.as_string())
            print(f"Email sent successfully to {receiver}")

        server.quit()

    def run(self):
        """Main pipeline execution"""

        risky_asteroids, summary, recipients = self.fetch_data()

        html_template = self.load_template()

        html_content = self.render_template(
            html_template,
            risky_asteroids,
            summary
        )

        recipient_emails = recipients["email"].tolist()

        self.send_emails(recipient_emails, html_content)


if __name__ == "__main__":
    alert_service = AlertEmailService()
    alert_service.run()