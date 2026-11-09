from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_elasticache as elasticache,
    aws_rds as rds,
)
from constructs import Construct


class DatabaseStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        rds_sg: ec2.ISecurityGroup,
        redis_sg: ec2.ISecurityGroup,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # RDS PostgreSQL
        # ------------------------------------------------------------------
        self.db_instance = rds.DatabaseInstance(
            self,
            "AiBallDb",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_15,
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE4_GRAVITON,
                ec2.InstanceSize.MICRO,
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
            ),
            security_groups=[rds_sg],
            database_name="aiball",
            credentials=rds.Credentials.from_generated_secret(
                username="aiball_admin",
            ),
            allocated_storage=20,
            storage_type=rds.StorageType.GP3,
            multi_az=False,
            deletion_protection=False,
            removal_policy=RemovalPolicy.SNAPSHOT,
            backup_retention=Duration.days(7),
        )

        # The auto-generated secret containing host, port, username, password
        self.db_secret = self.db_instance.secret

        # ------------------------------------------------------------------
        # ElastiCache Redis (L1 construct — no L2 available yet)
        # ------------------------------------------------------------------
        private_subnets = vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
        )

        redis_subnet_group = elasticache.CfnSubnetGroup(
            self,
            "RedisSubnetGroup",
            description="Subnet group for aiball Redis cluster",
            subnet_ids=private_subnets.subnet_ids,
            cache_subnet_group_name="aiball-redis-subnets",
        )

        redis_cluster = elasticache.CfnCacheCluster(
            self,
            "AiBallRedis",
            cluster_name="aiball-redis",
            engine="redis",
            engine_version="7.1",
            cache_node_type="cache.t4g.micro",
            num_cache_nodes=1,
            cache_subnet_group_name=redis_subnet_group.cache_subnet_group_name,
            vpc_security_group_ids=[redis_sg.security_group_id],
        )

        # Ensure the subnet group is created before the cluster
        redis_cluster.add_dependency(redis_subnet_group)

        self.redis_endpoint = redis_cluster.attr_redis_endpoint_address
        self.redis_port = redis_cluster.attr_redis_endpoint_port

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
        CfnOutput(self, "RdsEndpoint",
                  value=self.db_instance.db_instance_endpoint_address)
        CfnOutput(self, "RedisEndpoint",
                  value=self.redis_endpoint)
        CfnOutput(self, "RedisPort",
                  value=self.redis_port)
