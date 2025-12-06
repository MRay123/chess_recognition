# Instructions on how to run my code.

## First, run this command to install dependencies.

    ```bash
    pip install -r requirements.txt
    ```
    
## Next, if you want to run the final model, run this. It will run the model on 1000 test images and FEN numbers.

    ```bash
    python utils/evaluate_folder.py data/sample_boards --csv results.csv
    ```

## Afterwards, there is a few options for various things you wish to do.

## If you want to test on 20000 images, run this. WARNING: THIS WILL TAKE A VERY LONG TIME.

    ```bash
    python utils/evaluate_folder.py data/mass_boards --csv results.csv
    ```

## If running 1000 is too long, you can run on 100 boards with this command.

    ```bash
    python utils/evaluate_folder.py data/example_boards --csv results.csv
    ```

## If you desire to see a specific image and what that model sees each piece as, run this. I highly reccomend copy and pasting each image name.

    ```bash
    python predict.py
    ```
## You will need to past said image name into line 112 of predict.py