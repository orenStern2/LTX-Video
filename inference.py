from transformers import HfArgumentParser
import logging

from ltx_video.inference import infer, InferenceConfig


def main():
    # Set up logging to see more detailed error messages
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        print("Starting LTX-Video inference...")
        parser = HfArgumentParser(InferenceConfig)
        config = parser.parse_args_into_dataclasses()[0]
        print(f"Configuration loaded: {config}")
        print("Starting inference process...")
        
        # Add more detailed error handling around the infer call
        print("About to call infer function...")
        result = infer(config=config)
        print(f"Infer function returned: {result}")
        print("Inference completed successfully!")
        
    except KeyboardInterrupt:
        print("Inference was interrupted by user")
        raise
    except Exception as e:
        print(f"ERROR OCCURRED: {type(e).__name__}: {str(e)}")
        logging.error(f"Error during inference: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
