"""
Initialize database schema
"""

import logging
from polymarket.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database"""
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    main()
