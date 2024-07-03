# RefinPaint: Music Proofreading with RefinPaint

Code and models for: P. Ramoneda<sup>1+</sup>, M. Rocamora<sup>1</sup> & T. Akama<sup>2</sup>, “Music Proofreading with RefinPaint: Where and How to Modify Compositions given Context”, in Proc. of the 25th Int. Society for Music Information Retrieval Conf., San Francisco, USA, 2024.
<sup>1</sup> Pompeu Fabra University, Barcelona
<sup>2</sup> Sony Computer Science Laboratories. Tokyo
<sup>+</sup> Work carried out at Sony CSL, Tokyo

In RefinPaint user selects a MIDI section for enhancement(gray rectangle). Our methodology uses token-level feed-
back (blue) to highlight critical notes or sequences (red) for regeneration (green). This cycle repeats iteratively.


<object data="https://github.com/user-attachments/files/16089764/RefinPaint.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/user-attachments/files/16089764/RefinPaint.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf](https://github.com/user-attachments/files/16089764/RefinPaint.pdf">Download PDF</a>.</p>
    </embed>
</object>




## Abstract
Autoregressive generative transformers have become key players in music generation, known for their ability to produce coherent compositions. However, they often encounter challenges in human-machine collaboration, an area crucial for creative processes. To address this, we propose **RefinPaint**, an innovative iterative technique designed to enhance the sampling process used in music generation.

RefinPaint operates by identifying weaker elements within a musical composition through a dedicated feedback model. This feedback then guides an inpainting model in making more informed choices during the resampling process. This dual-focus methodology achieves two significant outcomes: it not only improves the machine's automatic inpainting generation capabilities through repeated refinement cycles but also provides a valuable tool for human composers. By integrating automatic proofreading into the composition process, RefinPaint helps in refining musical works with precision.

Our experimental results showcase RefinPaint's effectiveness in both inpainting and proofreading tasks, underlining its potential to enhance music created by both machines and human composers. This novel approach fosters creativity and supports amateur composers in enhancing their compositions, bridging the gap between human musicality and artificial intelligence.

>Note: everything is anonymous for review purposes.

>Supplementary code submitted to the 2024 International Conference on Music Information Retrieval (ISMIR 2024).

## Overview
RefinPaint is a cutting-edge tool designed to revolutionize music generation and editing. By facilitating an iterative refinement process, it aims to enhance the collaboration between humans and machines in the art of composition. This tool is perfect for composers, music producers, and anyone interested in exploring the boundaries of creative music generation.

### Code Structure
- `InpaintingModel.py`: Contains the inpainting model used for generating music fragments.
- `FeedbackModel.py`: Includes the feedback model for evaluating music elements and providing suggestions for improvement.
- `RefinPaint.py`: The main algorithm. It orchestrates the feedback and inpainting processes and interacts with the user through a command-line interface.
- `chekpoints/`: weights of the inpainting model and feedback models.

## Installation
### Prerequisites

- Python 3.6 or newer

- Pip



### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/anonymous/anonymous.git
   ```
   
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Install commandline musescore in linux:
   ```sh
   sudo apt-get install musescore
   ```

## Usage
To run RefinPaint, use the following command-line interface:

```bash
python RefinPaint.py --path <path_to_midi_file> --bar_begin <start_bar> --bar_end <end_bar> --confidence_about_your_composition <confidence_level> --human_in_the_loop --only_human <only_human>
```

## Command-Line Arguments
To run RefinPaint, use the following command-line interface:

```bash
python RefinPaint.py --path <path_to_midi_file> --bar_begin <start_bar> --bar_end <end_bar> --confidence_about_your_composition <confidence_level> --human_in_the_loop --only_human
```

- `--path`: Path to the MIDI file.
- `--bar_begin`: Starting bar number.
- `--bar_end`: Ending bar number.
- `--confidence_about_your_composition`: Confidence level about the composition (0-10).
- `--human_in_the_loop`: Include human in the loop processing (default: False). You can add red notes that you want to inpaint in the heatmap and then save the file in `proofreading` directory. Further instructions will be added after acceptance.
- `--only_human`: Use only human-generated compositions (default: False). 

For a detailed explanation of each argument, refer to the `parse_arguments` function within `RefinPaint.py`.

## Contributing
We welcome contributions from the community. If you're interested in improving RefinPaint or have suggestions, please follow our contribution guidelines. You can submit pull requests, report bugs, or suggest new features to help us improve this project.

## Citation

## FAQs
**Q: Can RefinPaint work with any music genre?**  
A: Yes, RefinPaint is designed to be genre-agnostic, making it suitable for a wide range of musical styles.

**Q: Is technical knowledge in music necessary to use RefinPaint?**  
A: No, RefinPaint is built to be accessible for users with varying levels of musical and technical expertise.

![sony](https://github.com/PRamoneda/RefinPaint/assets/16807496/d8106222-8ce1-4e95-ad2d-c551551f4d4a) ![mtg](https://github.com/PRamoneda/RefinPaint/assets/16807496/3ebc9341-82ce-4579-81b2-4b03f756c314)
