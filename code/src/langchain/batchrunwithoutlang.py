import pandas as pd
from transformers import pipeline

# Load your Excel file
excel_path = "data.csv"
df = pd.read_csv(excel_path)

print("✅ Excel data loaded:")
print(df)

# Initialize local lightweight model
generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

# Loop for user prompts
print("\n🔁 Ask me questions based on the Excel data (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("👋 Exiting.")
        break

    # Optional: inject Excel data into prompt
    context = df.to_string(index=False)
    full_prompt = f"Data:\n{context}\n\nQuestion: {user_input}\nAnswer:"

    # Generate answer
    response = generator(full_prompt, max_new_tokens=50, do_sample=True)
    answer = response[0]['generated_text'].split("Answer:")[-1].strip()

    print(f"🤖 {answer}\n")
