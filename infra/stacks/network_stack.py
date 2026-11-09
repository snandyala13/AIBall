from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
)
from constructs import Construct


class NetworkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # VPC: 2 AZs, public + private subnets, 1 NAT Gateway (~$32/mo)
        # Private subnets host RDS, ElastiCache, and Fargate tasks.
        # NAT Gateway lets private subnets reach external APIs.
        self.vpc = ec2.Vpc(
            self,
            "AiBallVpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # Security group for the ALB — allows inbound HTTPS from anywhere
        self.alb_sg = ec2.SecurityGroup(
            self,
            "AlbSg",
            vpc=self.vpc,
            description="ALB - inbound HTTPS from internet",
            allow_all_outbound=True,
        )
        self.alb_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(443),
            "Allow HTTPS from anywhere",
        )

        # Security group for Fargate tasks — only accepts traffic from the ALB
        self.fargate_sg = ec2.SecurityGroup(
            self,
            "FargateSg",
            vpc=self.vpc,
            description="Fargate tasks - inbound 8000 from ALB only",
            allow_all_outbound=True,
        )
        self.fargate_sg.add_ingress_rule(
            self.alb_sg,
            ec2.Port.tcp(8000),
            "Allow traffic from ALB on port 8000",
        )

        # Security group for EFS — only accepts NFS traffic from Fargate tasks
        self.efs_sg = ec2.SecurityGroup(
            self,
            "EfsSg",
            vpc=self.vpc,
            description="EFS - inbound NFS from Fargate only",
            allow_all_outbound=False,
        )
        self.efs_sg.add_ingress_rule(
            self.fargate_sg,
            ec2.Port.tcp(2049),
            "Allow NFS from Fargate tasks",
        )

        # Security group for RDS — only accepts PostgreSQL traffic from Fargate
        self.rds_sg = ec2.SecurityGroup(
            self,
            "RdsSg",
            vpc=self.vpc,
            description="RDS - inbound PostgreSQL from Fargate only",
            allow_all_outbound=False,
        )
        self.rds_sg.add_ingress_rule(
            self.fargate_sg,
            ec2.Port.tcp(5432),
            "Allow PostgreSQL from Fargate tasks",
        )

        # Security group for ElastiCache Redis — only accepts Redis traffic from Fargate
        self.redis_sg = ec2.SecurityGroup(
            self,
            "RedisSg",
            vpc=self.vpc,
            description="Redis - inbound 6379 from Fargate only",
            allow_all_outbound=False,
        )
        self.redis_sg.add_ingress_rule(
            self.fargate_sg,
            ec2.Port.tcp(6379),
            "Allow Redis from Fargate tasks",
        )
