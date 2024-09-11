import pytest
import os
from src.outputs.report_generate import create_final_report

@pytest.fixture
def realistic_data():
    """
    Fixture to create a realistic data dictionary for the report.
    """
    summary = """<h1>State of the Union Address 2024: Key Points</h1>

        <p>The President's State of the Union address highlighted various key points, including:</p>

        <h2>Economy and Jobs</h2>

        <ul>
        <li>The President emphasized that the economy is "the envy of the world" with 15 million new jobs created in three years.</li>
        <li>Unemployment is at a 50-year low, and a record 16 million Americans are starting small businesses.</li>
        <li>Historic job growth and small-business growth for Black, Hispanic, and Asian Americans.</li>
        <li>800,000 new manufacturing jobs in America and counting.</li>
        </ul>

        <h2>Education and Student Loans</h2>

        <ul>
        <li>The President wants to make college more affordable by increasing Pell Grants for working- and middle-class families.</li>
        <li>Record investments in HBCUs and minority-serving institutions, including Hispanic institutions.</li>
        <li>Fixed two student loan programs to reduce the burden of student debt for nearly 4 million Americans.</li>
        <li>Wants to give public school teachers a raise.</li>
        </ul>

        <h2>Healthcare and Prescription Drugs</h2>

        <ul>
        <li>Americans pay more for prescription drugs than anywhere in the world, which the President wants to change.</li>
        <li>Capped the cost of insulin at $35 a month for every American who needs it.</li>
        <li>Gave Medicare the power to negotiate lower prices on prescription drugs, saving seniors and taxpayers money.</li>
        <li>Protected and strengthened the Affordable Care Act, and wants to make tax credits for working families permanent.</li>
        </ul>

        <h2>Reproductive Freedom and Women's Rights</h2>

        <ul>
        <li>The President reiterated his support for reproductive freedom and the right to choose.</li>
        <li>Wants to restore Roe v. Wade as the law of the land.</li>
        <li>Guarantee the right to IVF nationwide.</li>
        </ul>

        <h2>Foreign Policy and Humanitarian Aid</h2>

        <ul>
        <li>The President emphasized the need for humanitarian assistance in Gaza.</li>
        <li>Directed the U.S. military to lead an emergency mission to establish a temporary pier in the Mediterranean to receive large shipments of aid.</li>
        <li>Called on Israel to allow more aid into Gaza and protect humanitarian workers.</li>
        </ul>

        <h2>Immigration and Border Security</h2>

        <ul>
        <li>The President wants to tackle the backlog of 2 million immigration cases.</li>
        <li>Wants to hire 1,500 more security agents and officers, 100 more immigration judges, and 4,300 more asylum officers.</li>
        <li>Proposed a bipartisan bill to bring order to the border and save lives.</li>
        </ul>

        <h2>Taxes and Fairness</h2>

        <ul>
        <li>The President wants to make the wealthy and big corporations pay their fair share.</li>
        <li>Proposed raising the corporate minimum tax to at least 21%.</li>
        <li>Wants to end tax breaks for Big Pharma, Big Oil, private jets, and massive executive pay.</li>
        </ul>
    """
    test_data = {
        'summary': summary,
        'chunk_words': [76, 79, 76, 75, 75, 75, 75, 90, 100, 75, 92, 124, 75, 76, 75, 98, 113, 87, 94, 90],
        'total_chunks': 123,
        'total_words': 10920,
        'total_tokens': 16895,
        'tokens_sent_tokens': 362,
        'labels': [9, 9, 9, 9, 4, 4, 4, 2, 2, 2, 2, 1, 3, 6, 6, 2, 3, 7, 0, 7, 7, 7, 7, 7, 8, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 1, 1, 1, 1, 1, 2, 2, 3, 6, 6, 6, 0, 6, 0, 6, 6, 6, 2, 0, 3, 3, 3, 3, 3, 0, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7,
          7, 7, 7, 3, 7, 7, 0, 0, 7, 8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 0, 8, 8, 8, 1, 8, 8, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2,
          3, 3, 2, 2, 6, 2, 2, 2, 2, 2, 9, 1],
        'themes': {0: 'Education and job accessibility.', 1: "Women's reproductive rights empowerment.", 
                   2: 'Unity and national optimism.', 3: 'Affordability of prescription drugs.',
                   4: 'Engagement metrics, unrelated content.', 5: 'Humanitarian aid in Gaza.', 
                   6: 'Economic revival through collaboration.', 7: 'Wealth inequality and taxation.', 
                   8: 'Immigration reform and security.', 9: 'US Government Administration'},
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