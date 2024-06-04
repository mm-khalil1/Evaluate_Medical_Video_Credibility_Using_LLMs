import pandas as pd
import os
import re
from langdetect import detect
'''
# Used first
QUESTION_HEAD = "You are a medical expert. Rate the following Transcript of a YouTube video according to this question:"
QUESTIONS = [
    # 1
    "Are the aims of the video clear?",
    # 2
    "Does the video achieve its aims?",
    # 3
    "Is the video relevant?",
    # 4
    "Is the video clear what sources of information were used to compile the transcript (other than the author)?",
    # 5
    "Is the video clear When the information used or reported was produced?",
    # 6
    "Is the video balanced and unbiased?",
    # 7
    "Does the video provide details of additional sources of support and information?",
    # 8
    "Does the video refer to areas of uncertainty?",
    # 9
    "Does the video describe how each treatment works?",
    # 10
    "Does the video describe the benefits of each treatment?",
    # 11
    "Does the video describe the risks of each treatment?",
    # 12
    "Does the video describe what would happen if no treatment is used?",
    # 13
    "Does the video describe how the treatment choices affect overall quality of life?",
    # 14
    "Is the video clear that there may be more than one possible treatment choice?",
    # 15
    "Does the video provide support for shared decision-making?",
]
QUESTION_TAIL = "Return an integer score from 1 to 5, where 1 means 'no', 3 means 'partially', and 5 means 'yes'. Then, explain your choice"
'''
QUESTION_HEAD = "You are a medical expert. Rate the following transcript of a YouTube video according to the given question."
QUESTIONS = [
    # 1
    """Are the aims of the video clear? Look for a clear indication of what it is about.
Return an integer score from 1 to 5.
Score of 5 means Yes: the video has clear aims.
Score 2 to 4 means Partially: the video has aims but they are unclear or incomplet.
Score of 1 means No: the video does not include any indication of its aims.""",
    # 2
    """Does the video achieve it's aims? Consider whether the video provides the information it aimed to outline.
Return an integer score from 1 to 5. 
Score of 5 means Yes: all the information you were expecting from a description of the aims has been provided.
Score of 2 to 4 means Partially: some of the information you were expecting from the aims has been provided.
Score of 1 means No - none of the information you were expecting from the aims has been provided.""",
    # 3
    """Is the video relevant? Consider whether the video addresses the questions that readers might ask.
Return an integer score from 1 to 5.
Score of 5 means Yes: the information is relevant.
Score 2 to 4 means Partially: the information is relevant to some extent.
Score of 1 means No: the information is not at all relevant.""",
    # 4
    """Is the video clear what sources of information were used to compile the video (other than the author)? Check whether the main claims or statements are accompanied by a reference to the sources used as evidence, e.g. a research study or expert opinion.
Return an integer score from 1 to 5.
Score of 5 means Yes: the sources of evidence are very clear.
Score 2 to 4 means Partially: the sources of evidence are clear to some extent. You may also give a partial rating to a video which quotes a reference in the text for some but not all of the main statements or facts.
Score of 1 means No: no sources of evidence for the information are mentioned.""",
    # 5
    """Is the video clear when the information used or reported was produced? Look for dates of the main sources of information used to compile the video.
Return an integer score from 1 to 5.
Score of 5 means Yes: dates for all acknowledged sources are clear.
Score 2 to 4 means Partially: only the date of the video itself is clear, or dates for some but not all acknowledged sources have been given.
Score of 1 means No: no dates have been given.""",
    # 6
    """Is the video balanced and unbiased? Look for a clear indication of whether the video is written from a personal or objective point of view, and evidence that a range of sources of information was used to compile the video, e.g. more than one research study or expert.
Be wary if: the video focuses on the advantages or disadvantages of one particular treatment choice without reference to other possible choices, the video relies primarily on evidence from single cases, or the information is presented in a sensational, emotive or alarmist way.
Return an integer score from 1 to 5.
Score of 5 means Yes: the information is completely balanced and unbiased.
Score 2 to 4 means Partially: some aspects of the information are unbalanced or biased.
Score of 1 means No: the information is completely unbalanced or biased.""",
    # 7
    """Does the video provide details of additional sources of support and information? Look for suggestions for further reading or for details of other organisations providing advice and information about the video topic.
Return an integer score from 1 to 5.
Score of 5 means Yes: the video provides full details of any additional source other than local branches of the same organisation.
Score 2 to 4 means Partially: the video provides details of an additional source or sources, but the details are incomplete or consist only of local branches of the same organisation.
Score of 1 means No: no additional sources of information are provided.""",
    # 8
    """Does the video refer to areas of uncertainty? Look for discussion of the gaps in knowledge or differences in expert opinion concerning treatment choices.
Return an integer score from 1 to 5.
Score of 5 means Yes: the video includes a clear reference to any uncertainty regarding treatment choices: this may be linked to each treatment choice or may be covered in a more general discussion or summary of the choices mentioned.
Score 2 to 4 means Partially: uncertainty is mentioned but the information is unclear or incomplete.
Score of 1 means No: no uncertainty about treatment choices is mentioned.""",
    # 9
    """Does the video describe how each treatment works? Look for a description of how a treatment acts on the body to achieve its effect.
Return an integer score from 1 to 5.
Score of 5 means Yes: the description of each treatment includes details of how it works.
Score 2 to 4 means Partially: the description of some but not all of the treatments includes details of how treatment works, or the details provided are unclear or incomplete.
Score of 1 means No: none of the descriptions about treatments include details of how treatment works.""",
    # 10
    """Does the video describe the benefits of each treatment? Benefits can include controlling or getting rid of symptoms, preventing recurrence of the condition and eliminating the condition, both short-term and long-term.
Return an integer score from 1 to 5.
Score of 5 means Yes: a benefit is described for each treatment.
Score 2 to 4 means Partially: a benefit is described for some but not all of the treatments.
Score of 1 means No: no benefits are described for any of the treatments.""",
    # 11
    """Does the video describe the risks of each treatment? Risks can include side-effects, complications and adverse reactions to treatment, both short-term and long-term.
Return an integer score from 1 to 5.
Score of 5 means Yes: a risk is described for each treatment.
Score 2 to 4 means Partially: a risk is described for some but not all of the treatments.
Score of 1 means No: no risks are described for any of the treatments.""",
    # 12
    """Does the video describe what would happen if no treatment is used? Look for a description of the risks and benefits of postponing treatment, of watchful waiting (i.e. monitoring how the condition progresses without treatment) or of permanently forgoing treatment.
Return an integer score from 1 to 5.
Score of 5 means Yes: there is a clear description of a risk or a benefit associated with any no treatment option.
Score 2 to 4 means Partially: a risk or benefit associated with a no treatment option is mentioned, but the information is unclear or incomplete.
Score of 1 means No: the video does not include any reference to the risks or benefits of no treatment options.""",
    # 13
    """Does the video describe how the treatment choices affect overall quality of life? Look for description of the effects of the treatment choices on day-to-day activity.
Return an integer score from 1 to 5.
Score of 5 means Yes: the video includes a clear reference to overall quality of life in relation to any of the treatment choices mentioned.
Score 2 to 4 means Partially: the video includes a reference to overall quality of life in relation to treatment choices, but the information is unclear or incomplete.
Score of 1 means No: there is no reference to overall quality of life in relation to treatment choices.""",
    # 14
    """Is the video clear that there may be more than one possible treatment choice? Look for a description of who is most likely to benefit from each treatment choice mentioned and under what circumstances, and look for suggestions of alternatives to consider or investigate further before deciding whether to select or reject a particular treatment choice.
Return an integer score from 1 to 5.
Score of 5 means Yes: the video makes it very clear that there may be more than one possible treatment choice.
Score 2 to 4 means Partially: the video indicates that there may be more than one possible choice, but the information is unclear or incomplete.
Score of 1 means No: the video does not give any indication that there may be a choice about treatment.""",
    # 15
    """Does the video provide support for shared decision-making? Look for suggestions of things to discuss with family, friends, doctors or other health professionals concerning treatment choices.
Return an integer score from 1 to 5.
Score of 5 means Yes: the video provides very good support for shared decision-making.
Score 2 to 4 means Partially: the video provides some support for shared decision-making.
Score of 1 means No: the video does not provide any support for shared decision-making.""",
]
QUESTION_TAIL = "Then explain you score in less than 220 words."

'''
# Binary questions for mDISCERN.
QUESTION_HEAD = "You are a medical expert. Rate the following transcript of a YouTube video according to the given question."
QUESTIONS = [
    # 1
    "Are the aims of the video clear and achieved?",
    # 2
    "Are reliable sources of information used? (ie, publication cited, speaker is a physician)",
    # 3
    "Is the information presented balanced and unbiased?",
    # 4
    "Are additional sources of information listed for patient reference?",
    # 5
    "Are areas of uncertainty mentioned?",
]
QUESTION_TAIL = "Return an integer score of either 1 (for yes) or 0 (for no). Then explain you choice in less than 220 words."

'''

def initialize_llm_ratings_df(videos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize DataFrame for LLM responses with 'None'.

    Args:
        videos_df (pandas.DataFrame): DataFrame containing video IDs and transcripts.

    Returns:
        pandas.DataFrame: DataFrame containing initialized LLM responses.
    """
    # Initialize DataFrame with 'Video ID' and 'Transcript' columns
    responses_df = pd.DataFrame({'Video ID': videos_df['Video ID'], 'Transcript': videos_df['Transcript']})
    # Columns 'Q1' to 'Q15' for ratings
    rating_columns = [f"Q{i}" for i in range(1, len(QUESTIONS) + 1)]
    responses_df[rating_columns] = None
    # Columns 'Response_1' to 'Response_15' for LLM responses
    response_columns = [f"Response_{i}" for i in range(1, len(QUESTIONS) + 1)]
    responses_df[response_columns] = None
    # Column 'Problem' if there is something wrong in the output
    responses_df['Problem'] = [set() for _ in range(len(videos_df))]
    return responses_df

def load_responses_df(transcripts_dir: str, 
                      transcripts_file_name: str, 
                      responses_dir: str, 
                      results_file_name: str) -> pd.DataFrame:
    """
    Load or initialize a DataFrame for LLM responses.

    Args:
        transcripts_dir (str): Directory containing transcripts.
        transcripts_file_name (str): Name of the transcripts file.
        responses_dir (str): Directory containing response files.
        results_file_name (str): Name of the LLM model.

    Returns:
        pandas.DataFrame: DataFrame containing (initialized) LLM responses.
    """
    files_in_directory = os.listdir(responses_dir)
    matching_files = [file for file in files_in_directory if results_file_name in file]
    if len(matching_files) > 1:
        raise ValueError(f"There is more than one file with the given results_file_name: '{results_file_name}'.\n\
                         Found files: {', '.join(matching_files)}")

    # Load existing response DataFrame
    if matching_files:
        file_path = os.path.join(responses_dir, matching_files[0])
        responses_df = pd.read_csv(file_path, encoding='utf-8')
        responses_df['Problem'] = responses_df['Problem'].apply(eval)       # convert string-formatted set to set-formatted
    else:
        transcripts_file_path = os.path.join(transcripts_dir, transcripts_file_name)
        # Load video IDs and transcripts from CSV (to save memory)
        videos_df = pd.read_csv(transcripts_file_path, usecols=['Video ID', 'Transcript'], encoding='utf-8')
        responses_df = initialize_llm_ratings_df(videos_df)          # Initialize DataFrame with None values

    return responses_df

def remove_prompt_from_response(response: str) -> str:
    """
    Remove the prompt from the beginning of the response text.

    Args:
        response (str): The original response text containing the prompt.

    Returns:
        str: The response text with the prompt removed.
    """
    phrase = "\nScore: "
    idx = response.find(phrase)  # Find the starting index of the phrase in the response
    if idx != -1:
        # Remove the prompt by slicing the response from the end of the phrase
        response = response[idx + len(phrase):]
    return response

def check_repetition(text: str, min_phrase_words=5, min_occurrences=2) -> bool:
    """
    Find repeated sentences longer than 'min_words' words, occurring more than 'min_occurrences' times in the given text.
    Some models are repeating some phrases in their output.    
    """
    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    phrase_counter = {}             # Initialize a dictionary to store occurrences of phrases
    for sentence in sentences:
        words = sentence.split()        # Tokenize the sentence into words
        
        # Generate all possible phrases longer than min_length words
        for i in range(len(words) - min_phrase_words + 1):
            phrase = ' '.join(words[i:i+min_phrase_words])
            # Update the counter for this phrase
            phrase_counter[phrase] = phrase_counter.get(phrase, 0) + 1
    
    # Filter phrases that occur more than min_occurrences times
    repeated_phrases = [phrase for phrase, count in phrase_counter.items() if count > min_occurrences]
    
    return bool(repeated_phrases)

def check_non_english_language(text: str) -> bool:
    """
    Check if the given text is in a language other than English.

    Args:
    text (str): The text to be checked.

    Returns:
    bool: True if the text is not in English, False otherwise.
    """
    # Check if the detected language is English
    return detect(text) != 'en'

def fill_Problem_column(df: pd.DataFrame, video_id: str, question_num: int) -> None:
    """Add a question number to the 'Problem' set for the specified video ID."""
    idx = df.index[df['Video ID'] == video_id].item()
    df.at[idx, 'Problem'].add(question_num)

def extract_rating(response: str, rating_scale) -> int:
    """
    Extract rating integer from beginning of LLM response.
    Note: Prompt should have been removed already from the beginning of the response.

    Returns:
        int or None: The extracted rating if found, otherwise None.
    """
    if rating_scale == 5:
        pattern = r'([1-5])'
    elif rating_scale == 1:
        pattern = r'([0-1])'
    else:
        raise ValueError("rating_scale should be either 5 or 1.")
    match = re.search(pattern, response)  # Search for the first encountered integer
    return int(match.group(1)) if match else None

def check_and_store_response(response: str,
                             responses_df: pd.DataFrame,
                             video_id: str,
                             question_num: int,
                             rating_scale: int = 5,
                             remove_prompt: bool = False,
                             print_response: bool = False
                             ) -> None:
    """
    Check and store the response and associated rating for a given video ID and question number.

    Args:
        response (str): The response text.
        responses_df (pd.DataFrame): DataFrame containing response data.
        video_id (str): ID of the video.
        question_num (int): Number of the question.
        rating_scale (int): The scale used for rating extraction. Defaults to 5.
        remove_prompt (bool): Whether to remove the prompt from the response. Defaults to False.
        print_response (bool): Whether to print the response. Defaults to False.
    """
    if remove_prompt:
        response = remove_prompt_from_response(response)

    if print_response:
        print(f"Q{question_num} response: {response}")# if question_num == 5 else None
    
    # Check if 'video_id' exists in the 'Video ID' column
    if video_id in responses_df['Video ID'].values:
        # Find the index of the row corresponding to the video ID
        idx = responses_df.index[responses_df['Video ID'] == video_id].tolist()[0]
    else:
        print("No matching video_id found for", video_id)  
        return  

    # Store response
    responses_df.at[idx, f'Response_{question_num}'] = response
    
    # Extract integer rating if it exists
    rating = extract_rating(response, rating_scale)
    if rating is not None:
        responses_df.at[idx, f'Q{question_num}'] = rating    # Store rating

    # check for repeated phrases as well
    if rating is None or check_repetition(response) or check_non_english_language(response):
        fill_Problem_column(responses_df, video_id, question_num)

def build_question_prompt(transcript: str, question_num: int) -> str:
    """
    Constructs a prompt combining question head, specific question, question tail, and transcript.

    Args:
        transcript (str): The transcript associated with the prompt.
        question_num (int): The number of the specific question.

    Returns:
        str: The constructed prompt.
    """
    return f"""{QUESTION_HEAD}
Question: {QUESTIONS[question_num - 1]}
{QUESTION_TAIL}

Transcript: {transcript}"""