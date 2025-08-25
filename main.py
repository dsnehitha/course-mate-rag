"""
Main entry point for CourseMate RAG Application CLI.
"""

import time
import argparse
from src.services.rag_service import RAGService
from src.utils.logger import logger


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_name", required=True, help="Course name (matches folder in course_materials)")
    args = parser.parse_args()
    course_name = args.course_name

    # Map course_name to folder
    course_folder = f"course_materials/{course_name}"
    extracted_audio = f"{course_folder}/extracted_audio"
    extracted_images = f"{course_folder}/extracted_images"
    extracted_transcripts = f"{course_folder}/extracted_transcripts"

    # Ensure folders exist
    import os
    for d in [course_folder, extracted_audio, extracted_images, extracted_transcripts]:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Starting CourseMate RAG Application for course: {course_name} (folder: {course_folder})")

    # Pass course_name to RAGService (update RAGService to accept and use this)
    rag_service = RAGService(course_name=course_name)
    rag_service.print_database_status()
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
            result = rag_service.generate_answer(query)
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