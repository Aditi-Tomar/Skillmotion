"""
MySQL Database Setup Script
Run this script to create all necessary tables for SkillMotion AI
"""

import asyncio
import aiomysql
import os
from dotenv import load_dotenv

load_dotenv()


async def setup_database():
    """Create all necessary tables"""

    # Connection parameters
    host = os.getenv("MYSQL_HOST", "localhost")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "skillmotion_ai")
    port = int(os.getenv("MYSQL_PORT", 3306))

    try:
        # Connect to MySQL
        conn = await aiomysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            db=database
        )

        cursor = await conn.cursor()

        print("Creating tables...")

        # Create user_profiles table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE NOT NULL,
                resume_data LONGTEXT,
                skills LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_user_id (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Created user_profiles table")

        # Create conversation_sessions table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                mode VARCHAR(50),
                is_active BOOLEAN DEFAULT TRUE,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_user_id (user_id),
                INDEX idx_is_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Created conversation_sessions table")

        # Create conversation_messages table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                message_type VARCHAR(50),
                content LONGTEXT,
                metadata JSON,
                was_interrupted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions(id) ON DELETE CASCADE,
                INDEX idx_session_id (session_id),
                INDEX idx_user_id (user_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Created conversation_messages table")

        # Create pending_queries table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_queries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                query LONGTEXT,
                priority INT DEFAULT 0,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP NULL,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions(id) ON DELETE CASCADE,
                INDEX idx_session_id (session_id),
                INDEX idx_status (status),
                INDEX idx_priority (priority)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Created pending_queries table")

        # Create support_requests table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_requests (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                subject VARCHAR(255),
                description LONGTEXT,
                context JSON,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_user_id (user_id),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Created support_requests table")

        await conn.commit()
        await cursor.close()
        conn.close()

        print("\n✅ Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Update your .env file with MySQL credentials:")
        print("   MYSQL_HOST=localhost")
        print("   MYSQL_USER=root")
        print("   MYSQL_PASSWORD=your_password")
        print("   MYSQL_DATABASE=skillmotion_ai")
        print("   MYSQL_PORT=3306")
        print("\n2. Restart the backend server")

    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(setup_database())
