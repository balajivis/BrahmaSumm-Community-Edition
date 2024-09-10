import pytest
import os
from src.outputs.report_generate import create_final_report

@pytest.fixture
def realistic_data():
    """
    Fixture to create a realistic data dictionary for the report.
    """
    test_data = {
        'summary': (
            "The government has outlined several initiatives and goals across various sectors to "
            "improve the lives of Americans and address pressing global issues. In education, the "
            "government aims to increase affordability by boosting Pell Grants and investing in "
            "minority-serving institutions. Additionally, they are connecting local businesses with "
            "high schools to provide students with hands-on experience and a pathway to well-paying jobs. "
            "This initiative is designed to enhance vocational training and create a more skilled workforce.\n"
            "Regarding reproductive rights, the government is committed to reinstating Roe v. Wade as "
            "the law of the land, ensuring reproductive freedom for all Americans. The government's "
            "confidence stems from their past successes in elections related to reproductive rights "
            "and their expectation of continued support in the 2024 elections.\n"
            "On healthcare, the government has taken steps to make healthcare more affordable, including "
            "reducing the cost of insulin for people with diabetes. They remain committed to ensuring that "
            "all Americans have access to affordable healthcare.\nInternationally, the government is working to "
            "provide humanitarian assistance to Gaza by establishing a temporary pier to facilitate the delivery "
            "of aid. They are also collaborating with Israel to ensure the safety of humanitarian workers.\n"
            "In terms of economic recovery, the government has supported the revival of the auto industry in "
            "Belvidere, Illinois, working with unions to keep the plant open and restore jobs for thousands of "
            "workers. Furthermore, the government is committed to protecting and strengthening Social Security, "
            "ensuring that the wealthy pay their fair share of taxes, and preventing corporations from raising "
            "prices to pad their profits.\nAnother key initiative is the government's effort to pass bipartisan "
            "immigration reform legislation. This legislation aims to hire more security agents and immigration "
            "judges, resolving asylum cases more efficiently.\nOverall, the government's goals are centered around "
            "unity, optimism, and collaboration with the American people to build a better future."
            
        ),
        'chunk_words': [76, 79, 76, 75, 75, 75, 75, 90, 100, 75, 92, 124, 75, 76, 75, 98, 113, 87, 94, 90],
        'total_chunks': 123,
        'total_words': 10920,
        'total_tokens': 16895,
        'tokens_sent_tokens': 362,
        'labels': [9, 9, 9, 9, 4, 4, 4, 2, 2, 2, 2, 1, 3, 6, 6, 2, 3, 7, 0, 7, 7, 7, 7, 7, 8, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 1, 1, 1, 1, 1, 2, 2, 3, 6, 6, 6, 0, 6, 0, 6, 6, 6, 2, 0, 3, 3, 3, 3, 3, 0, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7,
          7, 7, 7, 3, 7, 7, 0, 0, 7, 8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 0, 8, 8, 8, 1, 8, 8, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2,
          3, 3, 2, 2, 6, 2, 2, 2, 2, 2, 9, 1],
        'themes': {
            'Cluster 0': "Education and Job Opportunities",
            'Cluster 1': "Reproductive Rights",
            'Cluster 2': "Unity and Optimism",
            'Cluster 3': "Affordable Healthcare",
            'Cluster 4': "Irrelevant Data",
            'Cluster 5': "Humanitarian Assistance in Gaza",
            'Cluster 6': "Economic Recovery",
            'Cluster 7': "Taxation and Social Security",
            'Cluster 8': "Immigration Reform",
            'Cluster 9': "White House Website"
        },
        'umap_image_path': 'reports/umap_clusters.png'  # Assume the UMAP image already exists
    }
    return  test_data

def test_create_final_report(realistic_data):
    """
    Test to check if the PDF report is generated successfully in a persistent location.
    """
    # Step 1: Define the report path (set to a persistent location for manual inspection)
    report_path = 'reports/test_final_report.pdf'
    
    # Step 2: Call the create_final_report function
    create_final_report(realistic_data, report_path=report_path)
    
    # Step 3: Check if the PDF file was created
    assert os.path.exists(report_path), "The final report PDF should be generated."
    
    # Step 4: Open the generated PDF manually for visual inspection
    print(f"Generated PDF report saved at: {os.path.abspath(report_path)}")

# Run with pytest