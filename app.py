import streamlit as st
import requests
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Configuration
st.set_page_config(
    page_title="Assistant d'Achat Conversationnel",
    page_icon="🛒",
    layout="wide"
)

load_dotenv()

# --- Configuration API ---
@st.cache_data
def load_config():
    return {
        "openai_key": os.getenv("OPENAI_API_KEY"),
        "serp_key": os.getenv("SERPAPI_API_KEY")
    }

config = load_config()

def check_api_keys():
    if not config["openai_key"] or not config["serp_key"]:
        st.error("⚠️ Clés API manquantes (OPENAI_API_KEY, SERPAPI_API_KEY)")
        return False
    return True

if not check_api_keys():
    st.stop()

client = OpenAI(api_key=config["openai_key"])

# --- Système conversationnel ---
ASSISTANT_PERSONA = """
Tu es Alex, un expert en produits e-commerce passionné qui aide les gens à faire le bon choix d'achat.

TON RÔLE :
- Accompagner l'utilisateur comme un vendeur expert et bienveillant
- Poser les bonnes questions au bon moment, naturellement
- Vulgariser les caractéristiques techniques 
- Donner des conseils personnalisés basés sur l'usage réel
- Être conversationnel, pas robotique

APPROCHE :
1. Comprendre le besoin initial de façon naturelle
2. Poser 2-3 questions max par message pour creuser les besoins
3. Quand tu as assez d'infos, proposer une recherche de produits
4. Analyser chaque produit selon le profil utilisateur spécifique
5. Aider à la décision finale

STYLE :
- Conversationnel et chaleureux
- Questions courtes et naturelles  
- Explications simples sans jargon
- Emojis pour rendre vivant
- Toujours positif et encourageant

Tu DOIS maintenir un état des informations collectées sur l'utilisateur pour personnaliser tes conseils.
"""

def extract_price(price_str: str) -> float:
    if not price_str:
        return 0.0
    
    if isinstance(price_str, list):
        price_str = price_str[0] if price_str else "0"
    
    price_clean = re.sub(r'[^\d.,]', '', str(price_str))
    price_clean = price_clean.replace(',', '.')
    
    numbers = re.findall(r'\d+\.?\d*', price_clean)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return 0.0
    return 0.0

def format_delivery(delivery_info) -> str:
    if not delivery_info:
        return "Standard"
    
    if isinstance(delivery_info, list):
        return delivery_info[0][:30] if delivery_info else "Standard"
    
    return str(delivery_info)[:30]

@st.cache_data(ttl=1800)
def fetch_amazon_products(query: str, min_price: float = 0, max_price: float = 1000, num_results: int = 4) -> List[Dict]:
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "amazon",
        "amazon_domain": "amazon.fr",
        "api_key": config["serp_key"],
        "k": query,
        "s": "review-rank",
        "num": num_results * 2
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        products = []

        for product in data.get("organic_results", []):
            price_val = extract_price(product.get("price_str") or product.get("price", "0"))
            
            if min_price <= price_val <= max_price:
                rating = product.get("rating", 0)
                if isinstance(rating, str):
                    try:
                        rating = float(rating.split()[0])
                    except:
                        rating = 0
                
                reviews_count = product.get("ratings_total", 0)
                if isinstance(reviews_count, str):
                    reviews_count = int(re.sub(r'[^\d]', '', reviews_count) or 0)

                products.append({
                    "title": product.get("title", "Produit sans titre"),
                    "price": price_val,
                    "price_str": f"${price_val:.2f}" if price_val > 0 else "Prix non disponible",
                    "rating": rating,
                    "reviews_count": reviews_count,
                    "link": product.get("link", ""),
                    "description": product.get("snippet", "Description non disponible"),
                    "image": product.get("image", ""),
                    "delivery": product.get("delivery", ""),
                    "prime": product.get("prime", False)
                })

        products.sort(key=lambda x: (x["rating"], x["reviews_count"]), reverse=True)
        return products[:num_results]

    except Exception as e:
        st.error(f"⚠️ Erreur recherche : {str(e)}")
        return []

def chat_with_assistant(user_message: str, conversation_history: List[Dict], user_context: Dict) -> str:
    """Chat principal avec l'assistant qui gère tout le parcours d'achat."""
    
    # Construction du contexte
    context_info = f"""
    CONTEXTE UTILISATEUR ACTUEL :
    {json.dumps(user_context, indent=2, ensure_ascii=False)}
    
    HISTORIQUE RÉCENT :
    {json.dumps(conversation_history[-4:] if len(conversation_history) > 4 else conversation_history, indent=2, ensure_ascii=False)}
    """
    
    messages = [
        {"role": "system", "content": ASSISTANT_PERSONA + "\n\n" + context_info},
    ]
    
    # Ajouter l'historique récent
    for msg in conversation_history[-6:]:  # Garde les 6 derniers messages
        messages.append(msg)
    
    # Message actuel
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Désolé, j'ai un problème technique : {str(e)}"

def extract_search_intent(conversation_history: List[Dict]) -> Optional[Dict]:
    """Détecte si l'assistant veut faire une recherche produit."""
    
    if not conversation_history:
        return None
        
    last_message = conversation_history[-1].get('content', '')
    
    # Recherche de patterns indiquant une recherche
    search_patterns = [
        r"cherchons?\s+(.+?)(?:\s+entre|\s+dans|\s+à|\s+pour|\.|$)",
        r"recherche\s+(.+?)(?:\s+entre|\s+dans|\s+à|\s+pour|\.|$)",
        r"regardons?\s+(.+?)(?:\s+entre|\s+dans|\s+à|\s+pour|\.|$)"
    ]
    
    for pattern in search_patterns:
        match = re.search(pattern, last_message.lower())
        if match:
            query = match.group(1).strip()
            
            # Extraction du budget si mentionné
            budget_match = re.search(r'entre\s+(\d+)\s*(?:et|à|-)\s*(\d+)', last_message.lower())
            min_price = float(budget_match.group(1)) if budget_match else 0
            max_price = float(budget_match.group(2)) if budget_match else 1000
            
            return {
                "query": query,
                "min_price": min_price,
                "max_price": max_price
            }
    
    return None

def analyze_product_conversational(product: Dict, user_context: Dict) -> str:
    """Analyse un produit dans le style conversationnel."""
    
    prompt = f"""
    Tu es Alex, conseiller en achat. Présente ce produit à ton client en restant conversationnel :

    PRODUIT : {product['title']}
    Prix : {product['price_str']} | Note : {product['rating']}/5 | Avis : {product['reviews_count']}
    Description : {product['description']}

    PROFIL CLIENT : {json.dumps(user_context, ensure_ascii=False)}

    Donne ton avis en 2-3 phrases comme si tu parlais à un ami :
    - Pourquoi ça match avec ses besoins (ou pas)
    - Un point fort technique vulgarisé
    - Ton conseil final (je recommande / correct mais / à éviter)

    Reste naturel et direct, pas commercial.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except:
        return "Analyse en cours..."

# --- Interface principale ---
def main():
    st.title("🤖 Alex - Votre conseiller d'achat personnel")
    st.markdown("*Discutez avec moi comme avec un ami expert, je vous guide vers le produit parfait !*")
    
    # Initialisation de la session
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
        st.session_state.user_context = {}
        # Message de bienvenue
        welcome_msg = """
        Salut ! 👋 Je suis Alex, votre conseiller d'achat personnel.
        
        Je suis là pour vous aider à trouver exactement ce qu'il vous faut ! 
        
        Dites-moi simplement ce que vous cherchez et parlons-en ensemble. 
        Par exemple : "Je cherche un aspirateur" ou "J'ai besoin d'un casque pour le télétravail"
        
        Qu'est-ce que je peux vous aider à trouver aujourd'hui ? 😊
        """
        st.session_state.conversation.append({"role": "assistant", "content": welcome_msg})
    
    # Zone de conversation
    chat_container = st.container()
    
    with chat_container:
        # Affichage de l'historique
        for message in st.session_state.conversation:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
        
        # Vérification si recherche produit demandée
        search_intent = extract_search_intent(st.session_state.conversation)
        if search_intent and not st.session_state.get('products_shown', False):
            
            st.session_state.products_shown = True
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"🔍 Parfait ! Je cherche **{search_intent['query']}** pour vous...")
                
                products = fetch_amazon_products(
                    search_intent['query'], 
                    search_intent['min_price'], 
                    search_intent['max_price']
                )
                
                if products:
                    st.markdown(f"J'ai trouvé {len(products)} produits qui pourraient vous intéresser. Laissez-moi vous les présenter :")
                    
                    for i, product in enumerate(products, 1):
                        with st.expander(f"🏆 Option {i} : {product['title'][:70]}...", expanded=i == 1):
                            
                            col_main, col_image = st.columns([3, 1])
                            
                            with col_main:
                                # Infos essentielles
                                info_cols = st.columns(3)
                                with info_cols[0]:
                                    st.metric("💰 Prix", product['price_str'])
                                with info_cols[1]:
                                    st.metric("⭐ Note", f"{product['rating']:.1f}/5")
                                with info_cols[2]:
                                    st.metric("📝 Avis", f"{product['reviews_count']:,}")
                                
                                # Livraison
                                delivery_text = format_delivery(product.get('delivery'))
                                if product.get('prime'):
                                    st.markdown("🚚 **Prime** - Livraison gratuite rapide")
                                else:
                                    st.markdown(f"🚚 **Livraison :** {delivery_text}")
                                
                                # Analyse personnalisée
                                st.markdown("**🤖 Mon avis pour vous :**")
                                analysis = analyze_product_conversational(product, st.session_state.user_context)
                                st.info(analysis)
                                
                                # Lien Amazon
                                st.markdown(f"[🛒 **Voir sur Amazon**]({product['link']})")
                            
                            with col_image:
                                if product.get('image'):
                                    st.image(product['image'], width=180)
                    
                    # Invite à continuer la conversation
                    followup_msg = """
                    Voilà mes suggestions ! 😊
                    
                    Qu'est-ce que vous en pensez ? Avez-vous des questions sur l'un de ces produits ? 
                    Ou souhaitez-vous que je vous aide à affiner votre recherche ?
                    """
                    
                    st.markdown(followup_msg)
                    st.session_state.conversation.append({"role": "assistant", "content": f"Recherche effectuée pour '{search_intent['query']}' - {len(products)} produits trouvés"})
                
                else:
                    st.warning("😔 Je n'ai pas trouvé de produits dans cette gamme de prix.")
                    st.markdown("Voulez-vous que je cherche dans une autre gamme de prix ou avec d'autres termes ?")
    
    # Zone de saisie
    user_input = st.chat_input("Tapez votre message ici... 💬")
    
    if user_input:
        # Ajouter le message utilisateur
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Réinitialiser le flag des produits pour permettre de nouvelles recherches
        st.session_state.products_shown = False
        
        # Génération de la réponse
        with st.spinner("Alex réfléchit..."):
            response = chat_with_assistant(
                user_input, 
                st.session_state.conversation[:-1],  # Sans le dernier message
                st.session_state.user_context
            )
        
        # Mise à jour du contexte utilisateur via IA
        context_update = update_user_context(user_input, st.session_state.user_context)
        if context_update:
            st.session_state.user_context.update(context_update)
        
        # Ajouter la réponse
        st.session_state.conversation.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Sidebar avec contexte utilisateur
    with st.sidebar:
        st.header("🎯 Ce que je sais de vous")
        
        if st.session_state.user_context:
            for key, value in st.session_state.user_context.items():
                if value:
                    st.markdown(f"**{key.replace('_', ' ').title()} :** {value}")
        else:
            st.markdown("*Je découvre vos besoins au fur et à mesure de notre conversation...*")
        
        st.markdown("---")
        
        # Bouton reset
        if st.button("🔄 Nouvelle conversation"):
            for key in ['conversation', 'user_context', 'products_shown']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Exemples de conversation")
        st.markdown("""
        - "Je cherche un aspirateur"
        - "J'ai besoin d'un casque pour le télétravail"  
        - "Je veux changer de smartphone"
        - "Il me faut un ordinateur pour étudier"
        - "Vous avez des conseils pour un cadeau ?"
        """)

def update_user_context(user_message: str, current_context: Dict) -> Dict:
    """Met à jour le contexte utilisateur basé sur son message."""
    
    prompt = f"""
    Message utilisateur : "{user_message}"
    Contexte actuel : {json.dumps(current_context, ensure_ascii=False)}
    
    Extrait SEULEMENT les nouvelles informations concrètes du message utilisateur.
    Retourne un JSON avec les clés pertinentes. 
    
    Exemples de clés : produit_cherche, budget, usage, taille_logement, animaux, sensibilite_bruit, priorites, contraintes, marque_preferee, etc.
    
    Si aucune nouvelle info, retourne {{}}
    Ne répète pas les infos déjà connues.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        # Nettoyer la réponse pour extraire le JSON
        if result.startswith('```'):
            result = result.split('```')[1]
        if result.startswith('json'):
            result = result[4:]
        
        return json.loads(result)
    except:
        return {}

def chat_with_assistant(user_message: str, conversation_history: List[Dict], user_context: Dict) -> str:
    """Conversation principale avec l'assistant."""
    
    context_info = f"""
    INFORMATIONS COLLECTÉES SUR L'UTILISATEUR :
    {json.dumps(user_context, indent=2, ensure_ascii=False)}
    
    INSTRUCTIONS SPÉCIALES :
    - Si tu as assez d'informations pour faire une recherche, utilise ce format exact : "cherchons [terme de recherche] entre [prix_min] et [prix_max]"
    - Sinon, pose 1-2 questions naturelles pour mieux cerner les besoins
    - Vulgarise toujours les aspects techniques
    - Reste dans ton rôle d'Alex, conseiller sympa et expert
    """
    
    messages = [
        {"role": "system", "content": ASSISTANT_PERSONA + "\n\n" + context_info},
    ]
    
    # Historique récent
    for msg in conversation_history[-8:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Désolé, j'ai un petit souci technique... Pouvez-vous répéter ? 😅"

if __name__ == "__main__":
    main()