import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
import io
import base64
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Charges Locatives Commerciales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #FFC107;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.markdown("<h1 class='main-header'>Analyseur de Charges Locatives Commerciales</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Une solution basée sur l'IA pour l'analyse et la structuration de vos documents de charges</p>", unsafe_allow_html=True)

# Fonction pour extraire le texte d'un PDF via OCR
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
    
    try:
        # Convertir les pages du PDF en images
        images = convert_from_bytes(open(temp_pdf_path, 'rb').read())
        
        # Extraire le texte de chaque page
        text = ""
        for i, image in enumerate(images):
            text += f"\n--- PAGE {i+1} ---\n"
            text += pytesseract.image_to_string(image, lang='fra')
        
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte: {e}")
        return None
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

# Fonction pour classifier le type de document
def classify_document(text):
    if re.search(r"ANALYSE\s+DES\s+CHARGES\s+LOCATIVES\s+COMMERCIALES|Analyse\s+de\s+conformité", text, re.IGNORECASE):
        return "rapport_analyse"
    
    if re.search(r"RELEVE\s+INDIVIDUEL\s+DES\s+CHARGES\s+LOCATIVES", text, re.IGNORECASE):
        return "releve_individuel"
    
    if re.search(r"RELEVE\s+GENERAL\s+DE\s+DEPENSES", text, re.IGNORECASE):
        return "releve_general"
    
    if re.search(r"AVOIR|FACTURE", text, re.IGNORECASE):
        return "facture"
    
    return "document_inconnu"

# Fonction pour extraire les données d'un rapport d'analyse
def extract_rapport_analyse(text):
    data = {
        "type_document": "Rapport d'analyse des charges",
        "date_extraction": datetime.now().isoformat(),
        "metadata": {},
        "donnees": {
            "informations_generales": {},
            "charges": [],
            "charges_contestables": [],
            "recommandations": []
        }
    }
    
    # Extraire les métadonnées de base
    date_match = re.search(r"Rapport généré le (\d{2}/\d{2}/\d{4})", text)
    if date_match:
        data["metadata"]["date_rapport"] = date_match.group(1)
    
    type_bail_match = re.search(r"Type de bail\s+(\w+)", text)
    if type_bail_match:
        data["metadata"]["type_bail"] = type_bail_match.group(1)
        data["donnees"]["informations_generales"]["type_bail"] = type_bail_match.group(1)
    
    montant_match = re.search(r"Montant total des charges\s+(\d+[.,]\d{2}€)", text)
    if montant_match:
        data["metadata"]["montant_total"] = montant_match.group(1)
        data["donnees"]["informations_generales"]["montant_total"] = montant_match.group(1)
    
    taux_match = re.search(r"Taux de conformité\s+(\d+%)", text)
    if taux_match:
        data["metadata"]["taux_conformite"] = taux_match.group(1)
        data["donnees"]["informations_generales"]["taux_conformite"] = taux_match.group(1)
    
    # Extraire les charges
    charges_section = re.search(r"Analyse des charges facturées([\s\S]*?)Charges potentiellement contestables", text)
    if charges_section:
        charges_text = charges_section.group(1)
        charge_pattern = r"([^\n]+)\s+(\d+\.\d+)\s+(\d+\.\d+%)\s+(\w+)\s+(\w+)"
        for match in re.finditer(charge_pattern, charges_text):
            charge = {
                "poste": match.group(1).strip(),
                "montant": float(match.group(2)),
                "pourcentage": match.group(3),
                "conformite": match.group(4),
                "contestable": match.group(5) == "Oui"
            }
            data["donnees"]["charges"].append(charge)
    
    # Extraire les charges contestables
    contestable_section = re.search(r"Charges potentiellement contestables([\s\S]*?)Recommandations", text)
    if contestable_section:
        contestable_text = contestable_section.group(1)
        contestable_pattern = r"([^\n]+)\s*\((\d+\.\d+)€\)([\s\S]*?)Raison:\s*([^\n]*)([\s\S]*?)Justification:\s*([^\n]*)"
        for match in re.finditer(contestable_pattern, contestable_text):
            contestable = {
                "poste": match.group(1).strip(),
                "montant": float(match.group(2)),
                "raison": match.group(4).strip(),
                "justification": match.group(6).strip()
            }
            data["donnees"]["charges_contestables"].append(contestable)
    
    # Extraire les recommandations
    recommandations_section = re.search(r"Recommandations([\s\S]*?)Ce rapport a été généré", text)
    if recommandations_section:
        recommandations_text = recommandations_section.group(1)
        for line in recommandations_text.split('\n'):
            if line.strip() and re.match(r"\d+\.", line.strip()):
                data["donnees"]["recommandations"].append(re.sub(r"^\d+\.\s+", "", line.strip()))
    
    return data

# Fonction pour extraire les données d'un relevé individuel
def extract_releve_individuel(text):
    data = {
        "type_document": "Relevé individuel des charges locatives",
        "date_extraction": datetime.now().isoformat(),
        "metadata": {},
        "donnees": {
            "informations": {},
            "charges": [],
            "totaux": {}
        }
    }
    
    # Extraire les métadonnées de base
    periode_match = re.search(r"Période du (\d{2}/\d{2}/\d{4}) au (\d{2}/\d{2}/\d{4})", text)
    if periode_match:
        data["metadata"]["periode"] = f"{periode_match.group(1)} au {periode_match.group(2)}"
        data["donnees"]["informations"]["periode"] = f"{periode_match.group(1)} au {periode_match.group(2)}"
    
    societe_match = re.search(r"SCI PASTEUR", text)
    if societe_match:
        data["metadata"]["societe"] = "SCI PASTEUR"
    
    locataire_match = re.search(r"ELECTRO DEPOT FRANCE\s*\n\s*([^0-9]*)", text)
    if locataire_match:
        data["donnees"]["informations"]["locataire"] = "ELECTRO DEPOT FRANCE"
    
    bail_match = re.search(r"Bail : \d+ - Type : ([^\n]*)", text)
    if bail_match:
        data["donnees"]["informations"]["bail"] = bail_match.group(1).strip()
    
    # Extraire les charges
    charges_pattern = r"(\d{2})\s+([^\n]+)\s+(\d+[ \d]*,\d+)\s+(\d+)\s+(\d+,\d+)\s+(\d+)\s+(\d+[ \d]*,\d+)"
    for match in re.finditer(charges_pattern, text):
        charge = {
            "code": match.group(1),
            "designation": match.group(2).strip(),
            "montant": float(match.group(3).replace(" ", "").replace(",", ".")),
            "tantieme_global": int(match.group(4)),
            "tantieme_particulier": float(match.group(5).replace(",", ".")),
            "jours": int(match.group(6)),
            "quote_part": float(match.group(7).replace(" ", "").replace(",", "."))
        }
        data["donnees"]["charges"].append(charge)
    
    # Extraire les totaux
    total_charges_match = re.search(r"Total charges\s+(\d+[ \d]*,\d+)", text)
    if total_charges_match:
        data["donnees"]["totaux"]["total_charges"] = float(total_charges_match.group(1).replace(" ", "").replace(",", "."))
    
    provisions_match = re.search(r"Provisions\s+(-\d+[ \d]*,\d+)", text)
    if provisions_match:
        data["donnees"]["totaux"]["provisions"] = float(provisions_match.group(1).replace(" ", "").replace(",", "."))
    
    solde_match = re.search(r"Solde\s+(-?\d+[ \d]*,\d+)", text)
    if solde_match:
        data["donnees"]["totaux"]["solde"] = float(solde_match.group(1).replace(" ", "").replace(",", "."))
    
    return data

# Fonction pour extraire les données d'un relevé général
def extract_releve_general(text):
    data = {
        "type_document": "Relevé général de dépenses",
        "date_extraction": datetime.now().isoformat(),
        "metadata": {},
        "donnees": {
            "informations": {},
            "chapitres_charges": [],
            "total": None
        }
    }
    
    # Extraire les métadonnées de base
    periode_match = re.search(r"Période du (\d{2}/\d{2}/\d{4}) au (\d{2}/\d{2}/\d{4})", text)
    if periode_match:
        data["metadata"]["periode"] = f"{periode_match.group(1)} au {periode_match.group(2)}"
        data["donnees"]["informations"]["periode"] = f"{periode_match.group(1)} au {periode_match.group(2)}"
    
    societe_match = re.search(r"Société : (\d+)\s+([^\n]+)", text)
    if societe_match:
        data["metadata"]["societe"] = f"{societe_match.group(1)} {societe_match.group(2).strip()}"
    
    immeuble_match = re.search(r"Immeuble : (\d+)\s+([^\n]+)", text)
    if immeuble_match:
        data["donnees"]["informations"]["immeuble"] = f"{immeuble_match.group(1)} {immeuble_match.group(2).strip()}"
    
    # Extraire les chapitres de charges
    chapitre_pattern = r"Chap\.\s+(\d+)\s+([^\n]+)"
    for match in re.finditer(chapitre_pattern, text):
        chapitre = {
            "numero": match.group(1),
            "designation": match.group(2).strip(),
            "details": []
        }
        
        # Extraire les détails de chaque chapitre
        chapitre_text = re.search(f"Chap\\.\\s+{chapitre['numero']}\\s+{chapitre['designation']}([\\s\\S]*?)(?:Total Chap\\.\\s+{chapitre['numero']}|Chap\\.\\s+\\d+)", text)
        if chapitre_text:
            detail_pattern = r"([^\n]+)\s+(\d{2}/\d{2}/\d{4})\s+(\d+[ \d]*,\d+)\s+(\d+,\d+)\s+(\d+,\d+)"
            for detail_match in re.finditer(detail_pattern, chapitre_text.group(1)):
                detail = {
                    "designation": detail_match.group(1).strip(),
                    "date": detail_match.group(2),
                    "montant_ht": float(detail_match.group(3).replace(" ", "").replace(",", ".")),
                    "montant_tva": float(detail_match.group(4).replace(",", ".")),
                    "repartition": detail_match.group(5)
                }
                chapitre["details"].append(detail)
        
        data["donnees"]["chapitres_charges"].append(chapitre)
    
    # Extraire le total général
    total_match = re.search(r"Total clé 01\s+CHARGES COMMUNES\s+(\d+[ \d]*,\d+)", text)
    if total_match:
        data["donnees"]["total"] = float(total_match.group(1).replace(" ", "").replace(",", "."))
    
    return data

# Fonction principale pour analyser un document
def analyze_document(file_content, file_name):
    # Extraire le texte du document
    text = extract_text_from_pdf(file_content)
    
    if not text:
        return {"error": "Impossible d'extraire le texte du document"}
    
    # Classifier le type de document
    doc_type = classify_document(text)
    
    # Extraire les données selon le type de document
    if doc_type == "rapport_analyse":
        result = extract_rapport_analyse(text)
    elif doc_type == "releve_individuel":
        result = extract_releve_individuel(text)
    elif doc_type == "releve_general":
        result = extract_releve_general(text)
    else:
        result = {
            "type_document": "Document inconnu",
            "texte_extrait": text[:1000] + "..." if len(text) > 1000 else text
        }
    
    # Ajouter des informations sur le document source
    result["source"] = {
        "nom_fichier": file_name,
        "taille_fichier": len(file_content.getvalue())
    }
    
    return result

# Fonction pour créer un tableau de comparaison des charges
def create_comparison_chart(data_list):
    if len(data_list) < 2:
        st.warning("Il faut au moins deux documents pour effectuer une comparaison")
        return None
    
    # Préparer les données pour la comparaison
    comparison_data = []
    
    for data in data_list:
        period = data.get("metadata", {}).get("periode", "Période inconnue")
        
        if "donnees" in data and "charges" in data["donnees"]:
            for charge in data["donnees"]["charges"]:
                comparison_data.append({
                    "Période": period,
                    "Poste": charge.get("designation", charge.get("poste", "Inconnu")),
                    "Montant": charge.get("montant", 0)
                })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        return df
    else:
        return None

# Interface utilisateur Streamlit
st.sidebar.header("Options")
app_mode = st.sidebar.selectbox("Mode", ["Analyser un document", "Comparer des documents", "À propos"])

# Conserver les données analysées en session
if "analyzed_documents" not in st.session_state:
    st.session_state.analyzed_documents = []

# Mode d'analyse de document
if app_mode == "Analyser un document":
    st.markdown("<h2 class='sub-header'>Analyser un document</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Analyse du document en cours..."):
            # Réinitialiser le pointeur de fichier
            uploaded_file.seek(0)
            
            # Analyser le document
            result = analyze_document(uploaded_file, uploaded_file.name)
            
            # Ajouter le résultat à la session
            if "error" not in result:
                if result not in st.session_state.analyzed_documents:
                    st.session_state.analyzed_documents.append(result)
            
            # Afficher les résultats
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Document analysé avec succès: {result['type_document']}")
                
                # Afficher les informations générales
                st.markdown("<h3>Informations générales</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if "metadata" in result:
                        for key, value in result["metadata"].items():
                            st.write(f"**{key}:** {value}")
                
                # Afficher les données structurées selon le type de document
                if result["type_document"] == "Rapport d'analyse des charges":
                    # Afficher les charges
                    if "charges" in result["donnees"]:
                        st.markdown("<h3>Charges analysées</h3>", unsafe_allow_html=True)
                        charges_df = pd.DataFrame(result["donnees"]["charges"])
                        st.dataframe(charges_df)
                        
                        # Visualisation des charges
                        if not charges_df.empty:
                            st.markdown("<h3>Répartition des charges</h3>", unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            charges_df.plot.pie(
                                y='montant', 
                                labels=charges_df['poste'], 
                                autopct='%1.1f%%', 
                                ax=ax
                            )
                            st.pyplot(fig)
                    
                    # Afficher les charges contestables
                    if "charges_contestables" in result["donnees"] and result["donnees"]["charges_contestables"]:
                        st.markdown("<h3>Charges potentiellement contestables</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
                        
                        for charge in result["donnees"]["charges_contestables"]:
                            st.markdown(f"**{charge['poste']} ({charge['montant']}€)**", unsafe_allow_html=True)
                            st.markdown(f"Raison: {charge['raison']}", unsafe_allow_html=True)
                            st.markdown(f"Justification: {charge['justification']}", unsafe_allow_html=True)
                            st.markdown("---")
                        
                        st.markdown("    <p>Pour une version plus complète et personnalisée adaptée à vos besoins spécifiques, n'hésitez pas à nous contacter.</p>"
    </div>
    
    # Ajouter un espace pour les informations de déploiement
    st.markdown("---")
    st.markdown("### Déploiement sur Streamlit Cloud")
    
    with st.expander("Instructions de déploiement"):
        st.markdown("""
        Pour déployer cette application sur Streamlit Cloud :
        
        1. **Préparez votre environnement GitHub** :
           - Créez un nouveau dépôt GitHub
           - Téléchargez ce script et nommez-le `app.py`
           - Créez un fichier `requirements.txt` avec les dépendances suivantes :
        
        ```
        streamlit==1.27.0
        pandas==2.0.3
        numpy==1.24.3
        pillow==9.5.0
        pytesseract==0.3.10
        pdf2image==1.16.3
        matplotlib==3.7.2
        seaborn==0.12.2
        ```
        
        2. **Configurations supplémentaires** :
           - Pour l'OCR, ajoutez également les dépendances système dans un fichier `packages.txt` :
        
        ```
        tesseract-ocr
        tesseract-ocr-fra
        poppler-utils
        ```
        
        3. **Déploiement** :
           - Connectez-vous à [Streamlit Cloud](https://streamlit.io/cloud)
           - Créez une nouvelle application en pointant vers votre dépôt GitHub
           - Configurez les paramètres de déploiement en précisant que vous avez des dépendances système
           - Déployez l'application
        """)
    
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #888;">
    © 2025 Analyseur de Charges Locatives Commerciales<br>
    Version de démonstration 1.0.0
    </p>
    """, unsafe_allow_html=True)

# Exécuter l'application
if __name__ == "__main__":
    pass  # L'application Streamlit s'exécute automatiquement", unsafe_allow_html=True)
                    
                    # Afficher les recommandations
                    if "recommandations" in result["donnees"] and result["donnees"]["recommandations"]:
                        st.markdown("<h3>Recommandations</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        
                        for i, rec in enumerate(result["donnees"]["recommandations"], 1):
                            st.markdown(f"{i}. {rec}", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                elif result["type_document"] == "Relevé individuel des charges locatives":
                    # Afficher les informations
                    if "informations" in result["donnees"]:
                        st.markdown("<h3>Informations du relevé</h3>", unsafe_allow_html=True)
                        for key, value in result["donnees"]["informations"].items():
                            st.write(f"**{key}:** {value}")
                    
                    # Afficher les charges
                    if "charges" in result["donnees"]:
                        st.markdown("<h3>Charges facturées</h3>", unsafe_allow_html=True)
                        charges_df = pd.DataFrame(result["donnees"]["charges"])
                        st.dataframe(charges_df)
                        
                        # Visualisation des charges
                        if not charges_df.empty:
                            st.markdown("<h3>Répartition des charges</h3>", unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            charges_df.plot.bar(
                                x='designation', 
                                y='quote_part', 
                                ax=ax
                            )
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Afficher les totaux
                    if "totaux" in result["donnees"]:
                        st.markdown("<h3>Totaux</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        
                        for key, value in result["donnees"]["totaux"].items():
                            st.write(f"**{key}:** {value}€")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                elif result["type_document"] == "Relevé général de dépenses":
                    # Afficher les informations
                    if "informations" in result["donnees"]:
                        st.markdown("<h3>Informations du relevé</h3>", unsafe_allow_html=True)
                        for key, value in result["donnees"]["informations"].items():
                            st.write(f"**{key}:** {value}")
                    
                    # Afficher les chapitres de charges
                    if "chapitres_charges" in result["donnees"]:
                        st.markdown("<h3>Chapitres de charges</h3>", unsafe_allow_html=True)
                        
                        for chapitre in result["donnees"]["chapitres_charges"]:
                            with st.expander(f"Chapitre {chapitre['numero']} - {chapitre['designation']}"):
                                if chapitre["details"]:
                                    details_df = pd.DataFrame(chapitre["details"])
                                    st.dataframe(details_df)
                                else:
                                    st.write("Aucun détail disponible pour ce chapitre")
                    
                    # Afficher le total
                    if "total" in result["donnees"] and result["donnees"]["total"]:
                        st.markdown("<h3>Total</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.write(f"**Total général:** {result['donnees']['total']}€")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # JSON brut
                with st.expander("Voir les données JSON brutes"):
                    st.json(result)
                
                # Téléchargement du JSON
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="{uploaded_file.name.split(".")[0]}_analyse.json">Télécharger les résultats au format JSON</a>'
                st.markdown(href, unsafe_allow_html=True)

# Mode de comparaison de documents
elif app_mode == "Comparer des documents":
    st.markdown("<h2 class='sub-header'>Comparer des documents</h2>", unsafe_allow_html=True)
    
    if not st.session_state.analyzed_documents:
        st.warning("Aucun document analysé disponible pour la comparaison. Veuillez d'abord analyser des documents.")
    else:
        # Afficher la liste des documents analysés
        doc_options = [f"{doc['type_document']} - {doc.get('metadata', {}).get('periode', 'Période inconnue')} ({doc['source']['nom_fichier']})" for doc in st.session_state.analyzed_documents]
        selected_docs = st.multiselect("Sélectionner les documents à comparer", doc_options)
        
        if len(selected_docs) >= 2:
            # Récupérer les indices des documents sélectionnés
            selected_indices = [doc_options.index(doc) for doc in selected_docs]
            selected_data = [st.session_state.analyzed_documents[i] for i in selected_indices]
            
            # Créer la comparaison
            comparison_df = create_comparison_chart(selected_data)
            
            if comparison_df is not None:
                st.markdown("<h3>Tableau comparatif des charges</h3>", unsafe_allow_html=True)
                st.dataframe(comparison_df)
                
                # Visualisation de la comparaison
                st.markdown("<h3>Graphique comparatif</h3>", unsafe_allow_html=True)
                
                # Créer un tableau croisé dynamique pour la comparaison
                pivot_table = pd.pivot_table(
                    comparison_df,
                    values='Montant',
                    index='Poste',
                    columns='Période',
                    aggfunc='sum'
                )
                
                fig, ax = plt.subplots(figsize=(12, 8))
                pivot_table.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calcul de l'évolution globale
                if len(pivot_table.columns) >= 2:
                    st.markdown("<h3>Évolution globale</h3>", unsafe_allow_html=True)
                    
                    total_by_period = pivot_table.sum()
                    evolution_pct = []
                    
                    for i in range(1, len(total_by_period)):
                        prev = total_by_period.iloc[i-1]
                        curr = total_by_period.iloc[i]
                        evol = ((curr - prev) / prev) * 100
                        evolution_pct.append({
                            "De": total_by_period.index[i-1],
                            "À": total_by_period.index[i],
                            "Évolution (%)": round(evol, 2)
                        })
                    
                    if evolution_pct:
                        evolution_df = pd.DataFrame(evolution_pct)
                        st.dataframe(evolution_df)
                        
                        # Visualisation de l'évolution
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.bar(
                            [f"{e['De']} → {e['À']}" for e in evolution_pct],
                            [e['Évolution (%)'] for e in evolution_pct],
                            color=['green' if e['Évolution (%)'] <= 0 else 'red' for e in evolution_pct]
                        )
                        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        ax.set_ylabel('Évolution (%)')
                        ax.set_title('Évolution des charges entre périodes')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.warning("Impossible de créer une comparaison avec les documents sélectionnés")
        else:
            st.info("Veuillez sélectionner au moins deux documents pour effectuer une comparaison")

# Mode "À propos"
else:
    st.markdown("<h2 class='sub-header'>À propos de l'Analyseur de Charges Locatives Commerciales</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>L'<b>Analyseur de Charges Locatives Commerciales</b> est un outil développé pour faciliter l'analyse et la gestion des charges immobilières commerciales.</p>
    
    <p>Cet outil intègre des technologies d'intelligence artificielle et de reconnaissance optique de caractères (OCR) pour extraire automatiquement les informations pertinentes des documents de charges locatives, tels que :</p>
    <ul>
        <li>Rapports d'analyse de charges</li>
        <li>Relevés individuels de charges locatives</li>
        <li>Relevés généraux de dépenses</li>
        <li>Factures et avoirs</li>
    </ul>
    
    <h3>Fonctionnalités principales</h3>
    <ul>
        <li>Extraction automatique des données de vos documents PDF</li>
        <li>Classification du type de document</li>
        <li>Structuration des données en format JSON</li>
        <li>Visualisation des répartitions de charges</li>
        <li>Identification des charges potentiellement contestables</li>
        <li>Comparaison entre différentes périodes</li>
        <li>Analyse des évolutions de charges</li>
    </ul>
    
    <h3>Comment utiliser cet outil</h3>
    <ol>
        <li>Commencez par analyser vos documents dans l'onglet "Analyser un document"</li>
        <li>Une fois plusieurs documents analysés, utilisez l'onglet "Comparer des documents" pour visualiser leur évolution</li>
        <li>Vous pouvez télécharger les résultats au format JSON pour les intégrer à d'autres systèmes</li>
    </ol>
    
    <h3>Notes techniques</h3>
    <p>Cette application est une version de démonstration déployée sur Streamlit Cloud. Elle utilise :</p>
    <ul>
        <li>Streamlit pour l'interface utilisateur</li>
        <li>PyTesseract et pdf2image pour l'OCR</li>
        <li>Pandas et Matplotlib pour l'analyse de données et les visualisations</li>
        <li>RegEx pour l'extraction de patterns dans le texte</li>
    </ul>
