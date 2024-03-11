from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter
import tiktoken

with open('test_ja.txt', encoding='utf-8') as f:
    input_text = f.read()

bpe = tiktoken.encoding_for_model('gpt-4')

print("Token-based splitting")
text_splitter = TokenTextSplitter(model_name="gpt-4", chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_text(input_text)
for text in texts:
    print("Tokens={0}, Characters={1}, Text={2}".format(len(bpe.encode(text)), len(text), text))
    print("---")

print("Character-based splitting on paragraph boundary")

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
)

texts = text_splitter.split_text(input_text)
for text in texts:
    print("Tokens={0}, Characters={1}, Text={2}".format(len(bpe.encode(text)), len(text), text))
    print("---")

print("Recursive Character splitting")
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separators=[".", "。", "!", "！", "?", "？", ",", "、", "\n\n"],
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_text(input_text)
for text in texts:
    print("Tokens={0}, Characters={1}, Text={2}".format(len(bpe.encode(text)), len(text), text))
    print("---")


print("Token based recursive character splitting")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4", chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_text(input_text)
for text in texts:
    print("Tokens={0}, Characters={1}, Text={2}".format(len(bpe.encode(text)), len(text), text))
    print("---")
