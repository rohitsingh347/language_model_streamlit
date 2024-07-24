import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import pickle
from ngram_model import NGramLanguageModel, get_tokens, train_and_save_models, load_models, next_lines, preview_tokenizer, get_next_token

UNK, EOS = "_UNK_", "_EOS_"

# Load and preprocess data
data = pd.read_json("./arxivData.json")
lines = data.apply(lambda row: row['title'] + ' ; ' + row['summary'].replace("\n", ' '), axis=1).tolist()

# Define the directory to save models
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Train and save models if not already saved
if not all(os.path.exists(os.path.join(model_dir, f"{tokenizer_name}_model.pkl")) for tokenizer_name in ["WordPunctTokenizer", "word_tokenize", "BPE","Wordpiece"]):
    train_and_save_models(lines, model_dir)

# Load the models
models = load_models(model_dir)

# Function to highlight tokens

def highlight_tokens(sentence, tokens):
    highlighted_sentence = sentence
    for token in tokens:
        token_with_space = f' {token}'
        highlighted_sentence = highlighted_sentence.replace(token_with_space, f'<span style="background-color: #{random.randint(0, 0xFFFFFF):06x};">{token_with_space}</span>')
    return highlighted_sentence




def tokenizers_page():
    st.title("Tokenization")
    st.write("Select a tokenizer and input a sentence to see the tokenized output with highlights.")

    # Select tokenizer
    tokenizer_choice = st.radio("Choose Tokenizer", ["WordPunctTokenizer", "word_tokenize", "BPE", "SentencePiece", "Wordpiece"])

    # Preview tokenizer
    if st.button("Preview Tokenizer"):
        preview_image_path = preview_tokenizer(tokenizer_choice)
        st.image(preview_image_path, caption=f"{tokenizer_choice} Preview")

    # User input
    user_input = st.text_input("Enter a sentence to tokenize:")

    # Display tokenized output with highlights
    if user_input:
        tokens = get_tokens(tokenizer_choice, [user_input])[0]
        highlighted_sentence = highlight_tokens(user_input, tokens)
        st.markdown(f"Tokenized Sentence (using {tokenizer_choice}):")
        st.markdown(highlighted_sentence, unsafe_allow_html=True)


# Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Tokenization", "N-gram Language Model", 'Tokenization_page'])

if page == "N-gram Language Model":
    st.title("N-gram Language Model")

    # Select tokenizer
    tokenizer_name = st.radio("Select a Tokenizer", ["WordPunctTokenizer", "word_tokenize", "BPE","Wordpiece"])

    # User input
    user_input = st.text_input("Enter a word or sentence:")

    # Variables to store the state
    if 'top_five_words' not in st.session_state:
        st.session_state['top_five_words'] = []

    if 'initial_generated_sentence' not in st.session_state:
        st.session_state['initial_generated_sentence'] = ""

    if 'initial_top_five' not in st.session_state:
        st.session_state['initial_top_five'] = []

    if 'show_initial_chart' not in st.session_state:
        st.session_state['show_initial_chart'] = True

    # Button to generate and display sentence with bar chart
    if st.button('Generate Sentence and Show Chart'):
        if user_input:
            st.write(f"Input: {user_input} (using {tokenizer_name})")
            tokenized_input = ' '.join(get_tokens(tokenizer_name, [user_input.lower()])[0])
            lm = models[tokenizer_name]
            generated_sentence, top_five_words = next_lines(tokenized_input, lm, use_highest_prob=True)
            st.session_state['initial_generated_sentence'] = generated_sentence
            st.session_state['top_five_words'] = top_five_words
            
            # Display bar chart for the first prediction
            possible_next_tokens = lm.get_possible_next_tokens(tokenized_input)
            top_five = possible_next_tokens.most_common(5)
            words = [item[0] for item in top_five]
            probabilities = [item[1] for item in top_five]
            
            fig, ax = plt.subplots()
            ax.bar(words, probabilities, color='maroon', width=0.4)
            ax.set_xlabel("Top 5 Words")
            ax.set_ylabel("Probabilities")
            ax.set_title("Words with Probability")
            st.pyplot(fig)

    # Display initial generated sentence and bar chart if they exist
    if st.session_state['initial_generated_sentence']:
        st.write(f"Generated Sentence (using {tokenizer_name}): {st.session_state['initial_generated_sentence']}")
        top_five = st.session_state['initial_top_five']
        words = [item[0] for item in top_five]
        probabilities = [item[1] for item in top_five]
        
        if st.session_state['show_initial_chart']:
            fig, ax = plt.subplots()
            ax.bar(words, probabilities, color='maroon', width=0.4)
            ax.set_xlabel("Top 5 Words")
            ax.set_ylabel("Probabilities")
            ax.set_title("Words with Probability")
            st.pyplot(fig)

    # Button to sample and predict again without showing chart
    if st.button('Sample and Predict Again'):
        if st.session_state['initial_generated_sentence']:
            new_input = st.session_state['initial_generated_sentence']
            tokenized_input = ' '.join(get_tokens(tokenizer_name, [new_input.lower()])[0])
            lm = models[tokenizer_name]
            selected_token, top_five_words = get_next_token(lm, tokenized_input, use_highest_prob=False)
            new_generated_sentence = new_input + ' ' + selected_token
            st.session_state['initial_generated_sentence'] = new_generated_sentence  # Update the sentence
            st.session_state['top_five_words'] = top_five_words  # Update top five words
            st.session_state['show_initial_chart'] = False  # Hide initial chart
            
            st.write(f"Re-sampled Generated Sentence (using {tokenizer_name}): {new_generated_sentence}")

            # Display bar chart for the new prediction
            possible_next_tokens = lm.get_possible_next_tokens(tokenized_input)
            top_five = possible_next_tokens.most_common(5)
            words = [item[0] for item in top_five]
            probabilities = [item[1] for item in top_five]
            
            fig, ax = plt.subplots()
            ax.bar(words, probabilities, color='maroon', width=0.4)
            ax.set_xlabel("Top 5 Words")
            ax.set_ylabel("Probabilities")
            ax.set_title("Words with Probability")
            st.pyplot(fig)
elif page == 'Tokenization_page':
    tokenizers_page()
elif page == "Tokenization":
    st.title("Tokenizers")

    # Select tokenizer
    tokenizer_name = st.radio("Select a Tokenizer", ["WordPunctTokenizer", "word_tokenize", "BPE","Wordpiece"])

    # Preview selected tokenizer
    if st.button("Preview Tokenizer"):
        preview_image_path = preview_tokenizer(tokenizer_name)
        st.image(preview_image_path, caption=f"{tokenizer_name} Preview")

    # User input
    sentence = st.text_input("Enter a sentence to tokenize:")

    if st.button("Tokenize"):
        if sentence:
            tokens = get_tokens(tokenizer_name, [sentence])
            st.write(f"Tokenized Sentence (using {tokenizer_name}): {tokens[0]}")
        else:
            st.write("Please enter a sentence to tokenize.")
