import streamlit as st                  # pip install streamlit
from PIL import Image

st.markdown("# Structure-Based Classification of Antigen Presentation and T Cell Recognition")

#############################################

st.markdown("### Melissa Pathil | PAML Final Project | May 2025")

#############################################

st.markdown('<div style="text-align: justify;">T cell activation is triggered by the binding of T cell \
    receptors (TCR) to peptide antigens presented by major histocompatibility complex (MHC) proteins. \
    Due to the critical role T cells play in recognizing and responding to infection, significant efforts \
    have been made to understand and predict TCR specificity for peptide-MHC (pMHC) complexes. This project \
    aims to utilize structure-based metrics of pMHC-TCR complexes to accurately predict MHC presentation \
    and TCR recognition. We will train models using naïve Bayes classification and support vector \
    machines and evaluate performance using accuracy, precision, recall, and F1 score. By including \
    structural information in model training, we expect strong performance with MHC presentation and \
    improved performance for TCR recognition compared to existing sequence-based models. Regarding \
    impact, our project will provide insight into the effectiveness of structure-based features for \
    predicting TCR specificity.</div>', unsafe_allow_html=True)

pmhc_tcr_image_path = 'images\pMHC_TCR.png'
image = Image.open(pmhc_tcr_image_path)

st.image(image, caption='Peptide antigen presentation by MHC and recognition by \
    TCR by Blum et al. (left). Experimental pMHC-TCR structure from Protein Data Bank 6EQA (right).')

st.write("**Dataset:**  For our AlphaFold3-generated samples of pMHC-TCR complexes, we extracted \
 (1) AlphaFold3-predicted scores indicating local and global \
 confidence in the relative position and orientation of the \
 peptide, MHC, and TCR and (2) root mean  square difference (RMSD) for all mutant samples \
 relative to their corresponding wild-type peptide.")