from aa_utilities.wrappers import watermark as aa_watermark

output_str = aa_watermark(
    author="Amin Allahyar",
    email='Amin.Allahyar@astrazeneca.com',
    timezone='Europe/Stockholm',
    namespace=globals(),
)

print(output_str)
