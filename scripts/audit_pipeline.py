from pathlib import Path
from paper2lab.inference.pipeline import PaperPipeline

p = PaperPipeline(refinement_mode="nemotron")

total = 0
errors = 0
weak = 0

for pdf in Path("Data/papers").rglob("*.pdf"):
    total += 1

    try:
        r = p.run(str(pdf))
        c = r["paper_card_final"]

        roadmap = c.get("reproduction_roadmap") or {}
        if not isinstance(roadmap, dict):
            roadmap = {}

        kit = c.get("lab_starter_kit") or {}
        if not isinstance(kit, dict):
            kit = {}

        datasets = c.get("datasets_or_data_sources") or []
        findings = c.get("key_findings") or []
        roadmap_steps = roadmap.get("experimental_steps") or []
        kit_structure = kit.get("project_structure") or []
        kit_risks = kit.get("reproducibility_risks") or []

        is_weak = (
            not c.get("title")
            or not c.get("paper_type")
            or not roadmap
            or not kit
            or len(kit_structure) == 0
        )

        if is_weak:
            weak += 1

        print("\n---", pdf.name)
        print("type:", c.get("paper_type"))
        print("datasets:", len(datasets), datasets[:3])
        print("findings:", len(findings))
        print("roadmap steps:", len(roadmap_steps))
        print("lab kit structure:", len(kit_structure))
        print("lab risks:", len(kit_risks))
        print("refinement:", r["refinement"]["status"])
        print("weak:", is_weak)

    except Exception as e:
        errors += 1
        print("\nERROR:", pdf.name)
        print(str(e))

print("\n====================")
print("TOTAL:", total)
print("ERRORS:", errors)
print("WEAK:", weak)
print("====================")