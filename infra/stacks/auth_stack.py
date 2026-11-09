from aws_cdk import (
    Aws,
    CfnOutput,
    Stack,
    aws_cognito as cognito,
)
from constructs import Construct


class AuthStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        callback_urls = self.node.try_get_context("callback_urls") or [
            "http://localhost:8000/oauth2/idpresponse",
        ]

        # ------------------------------------------------------------------
        # Cognito User Pool
        # ------------------------------------------------------------------
        self.user_pool = cognito.UserPool(
            self,
            "AiBallUserPool",
            user_pool_name="aiball-users",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_uppercase=True,
                require_lowercase=True,
                require_digits=True,
                require_symbols=False,
            ),
            mfa=cognito.Mfa.OPTIONAL,
            mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
            ),
        )

        # ------------------------------------------------------------------
        # Cognito User Pool Client
        # ------------------------------------------------------------------
        self.user_pool_client = self.user_pool.add_client(
            "AiBallWebClient",
            user_pool_client_name="aiball-web-client",
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True,
            ),
            generate_secret=True,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL,
                        cognito.OAuthScope.PROFILE],
                callback_urls=callback_urls,
            ),
        )

        # ------------------------------------------------------------------
        # Cognito User Pool Domain (Cognito-hosted)
        # ------------------------------------------------------------------
        self.user_pool_domain = self.user_pool.add_domain(
            "AiBallUserPoolDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=f"aiball-{Aws.ACCOUNT_ID}",
            ),
        )

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
        CfnOutput(self, "UserPoolId",
                  value=self.user_pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId",
                  value=self.user_pool_client.user_pool_client_id)
        CfnOutput(
            self,
            "CognitoLoginUrl",
            value=(
                f"https://{self.user_pool_domain.domain_name}"
                f".auth.{Aws.REGION}.amazoncognito.com/login"
                f"?client_id={self.user_pool_client.user_pool_client_id}"
                f"&response_type=code"
                f"&redirect_uri={callback_urls[0]}"
            ),
        )
