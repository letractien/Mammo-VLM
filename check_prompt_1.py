def generate_mammogram_description(
    laterality,
    view_position,
    breast_density,
    breast_birads,
    finding_categories,
    finding_birads,
    width,
    height,
    xmin,
    ymin,
    xmax,
    ymax
):
    density_descriptions = {
        "DENSITY A": "乳房几乎完全由脂肪组织组成。",
        "DENSITY B": "乳房中散布有部分致密组织。",
        "DENSITY C": "乳腺组织密度相对较高。",
        "DENSITY D": "乳腺组织非常致密。"
    }

    birads_descriptions = {
        "BI-RADS 0": "需要进一步成像检查。",
        "BI-RADS 1": "未见明显异常。",
        "BI-RADS 2": "发现良性病变。",
        "BI-RADS 3": "存在轻微但不明确的异常。",
        "BI-RADS 4": "发现可疑病变，需要进一步评估。",
        "BI-RADS 5": "高度怀疑恶性病变。",
        "BI-RADS 6": "已确诊的恶性病变。"
    }

    category_descriptions = {
        "Mass": (
            "肿块（Mass）是可以在乳腺X线片上观察到的病变。"
            "它们通常呈圆形、椭圆形或不规则形状，边缘可能清晰、模糊或呈分叶状。"
        ),
        "Suspicious Calcification": (
            "可疑钙化灶（Suspicious Calcification）是细小、不规则的钙化点，"
            "通常成簇或线性分布，可能呈树枝状或形态不一致。"
        )
    }

    view_text = '头尾向（CC）' if view_position == 'CC' else '内外斜向（MLO）'
    side_text = '左侧' if laterality == 'L' else '右侧'
    density_description = density_descriptions.get(breast_density, "密度未明确。")
    breast_birads_desc = birads_descriptions.get(breast_birads, "未定义。")
    finding_birads_desc = birads_descriptions.get(finding_birads, "未定义。")

    if isinstance(finding_categories, list):
        finding_cat = finding_categories[0]
    elif isinstance(finding_categories, str):
        finding_cat = finding_categories.strip("[]'\" ")
    else:
        finding_cat = str(finding_categories)

    category_description = category_descriptions.get(finding_cat, "未识别的病变类型。")
    description = f"""该乳腺X线图像为{side_text}乳房（Laterality: {laterality}），拍摄视角为 {view_position}（{view_text}）。乳腺密度：{breast_density} – {density_description} 总体 BI-RADS 等级：{breast_birads} – {breast_birads_desc} 检测到的病变类型：{finding_cat} – {category_description} 病变 BI-RADS 等级：{finding_birads} – {finding_birads_desc} 图像尺寸：{width} × {height} 像素。病变框选区域从 ({xmin:.2f}, {ymin:.2f}) 到 ({xmax:.2f}, {ymax:.2f})。"""
    return description


print(generate_mammogram_description(
    laterality="L",
    view_position="CC",
    breast_density="DENSITY C",
    breast_birads="BI-RADS 3",
    finding_categories=["Mass"],
    finding_birads="BI-RADS 4",
    width=2800,
    height=3500,
    xmin=400,
    ymin=1500,
    xmax=800,
    ymax=2000
))
