"""
Client email template service
Send verification codes and other emails using SMTP
"""
from app.services.common.email_smtp import (
    send_verification_email,
    send_password_reset_email,
    send_welcome_email,
    send_waiting_list_verification_email,
    send_waiting_list_admin_notification
)


class ClientEmailService:
    """客户端邮件服务，封装注册验证、密码重置、欢迎邮件和等待列表通知的发送逻辑。"""

    async def send_registration_verification(self, email: str, verification_code: str, user_name: str = None):
        """
        Send registration verification code email

        Args:
            email: Recipient email
            verification_code: Verification code
            user_name: User name
        """
        await send_verification_email(
            email=email,
            verification_code=verification_code,
            first_name=user_name or "User",
            expires_minutes=10
        )

    async def send_password_reset_code(self, email: str, verification_code: str, user_name: str = None):
        """
        Send password reset verification code email

        Args:
            email: Recipient email
            verification_code: Verification code
            user_name: User name
        """
        await send_password_reset_email(
            email=email,
            verification_code=verification_code,
            first_name=user_name or "User",
            expires_minutes=10
        )

    async def send_welcome(self, email: str, user_name: str = None, dashboard_url: str = "https://prepwise.com/dashboard"):
        """
        Send welcome email

        Args:
            email: Recipient email
            user_name: User name
            dashboard_url: Dashboard URL
        """
        await send_welcome_email(
            email=email,
            first_name=user_name or "User",
            dashboard_url=dashboard_url
        )

    async def send_waiting_list_verification(self, email: str, verification_token: str, first_name: str = None):
        """
        Send waiting list verification email

        Args:
            email: Recipient email
            verification_token: JWT verification token
            first_name: User name
        """
        await send_waiting_list_verification_email(
            email=email,
            verification_token=verification_token,
            first_name=first_name or "User"
        )

    async def send_waiting_list_admin_notification(self, submission_data: dict):
        """
        Send waiting list submission notification to admin

        Args:
            submission_data: Submission data
        """
        await send_waiting_list_admin_notification(submission_data)


client_email_service = ClientEmailService()