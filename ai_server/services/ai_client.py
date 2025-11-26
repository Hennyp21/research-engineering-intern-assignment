# ai-server/services/ai_client.py
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import json
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def summarize_posts_with_openai(sample_texts: List[str], prompt_template: str) -> dict:
    """
    If OPENAI_KEY is present, call OpenAI. Otherwise return a placeholder.
    """
    if not OPENAI_KEY:
        # Return a placeholder and the prompt that should be used
        return {
            "summary": "OpenAI API key not configured. Put your key in ai-server/.env and restart.",
            "examples": [],
            "confidence": "low",
            "prompt_used": prompt_template + "\n\n" + "\n\n".join(sample_texts[:5])
        }

  
    api_key = os.getenv("OPENAI_API_KEY")   
    client = OpenAI(api_key=api_key)

    # 2. Prepare Data
    combined_text = "\n\n".join(sample_texts)
    
    # Safety truncation to prevent token errors (approx 12k chars)
    if len(combined_text) > 1200:
        combined_text = combined_text[:1200] + "...(truncated)"

    try:
        
        # 3. Call API (New Syntax: client.chat.completions.create)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": f"{prompt_template}\n\nData:\n{combined_text}"}
            ],
            temperature=0.5,
        )
        
        content = completion.choices[0].message.content.strip()

        # 4. Attempt to parse JSON response
        try:
            # Clean up markdown code blocks if AI adds them
            clean_content = content
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            
            return json.loads(clean_content.strip())
        except json.JSONDecodeError:
            # If AI didn't return valid JSON, return the raw string
            return content

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return {"summary": f"Error communicating with AI: {str(e)}"}