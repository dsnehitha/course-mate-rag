"""
Main entry point for CourseMate RAG Application CLI.
"""

import time
from src.services.rag_service import RAGService
from src.utils.logger import logger


def main():
    """Main CLI function."""
    logger.info("Starting CourseMate RAG Application...")
    
    # Initialize RAG service
    rag_service = RAGService()
    
    # Build collection
    logger.info("Building collection...")
    rag_service.build_collection()
    
    # Interactive query loop
    while True:
        try:
            query = input("\nEnter your query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting application...")
                break
            
            if not query:
                continue
            
            logger.info(f"Processing query: {query}")
            start_time = time.time()
            
            # Generate answer
            result = rag_service.generate_answer(query)
            
            # Print results
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {', '.join(result['sources'])}")
            
            if result['images']:
                print(f"\nImages used: {', '.join(result['images'])}")
            
            print(f"\nQuery processed in {time.time() - start_time:.2f} seconds.")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main() 