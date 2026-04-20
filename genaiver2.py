if active == "overview":
    # ── encode logo ──
    import base64, os
    logo_path = "msu_logo.png"   # put the logo file in the same folder as your script
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        logo_html = f"<img src='data:image/png;base64,{logo_b64}' style='height:90px; margin-bottom:16px; filter:brightness(0) invert(1);'/>"
    else:
        logo_html = ""

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C["ink"]} 0%,{C["mid"]} 100%);
                border-radius:12px; padding:40px 48px; color:white; margin-bottom:32px; text-align:center;'>
        {logo_html}
        <div style='font-size:13px; font-weight:700; text-transform:uppercase;
                    letter-spacing:2px; color:{C["teal_lt"]}; margin-bottom:6px;'>
            THE MAHARAJA SAYAJIRAO UNIVERSITY OF BARODA
        </div>
        <div style='font-size:12px; color:#94b4cc; margin-bottom:4px;'>Faculty of Science · Department of Statistics</div>
        <div style='font-size:12px; color:#94b4cc; margin-bottom:20px;'>Academic Year 2025-26</div>
        <div style='font-family:"Libre Baskerville",serif; font-size:28px;
                    font-weight:700; line-height:1.35; margin-bottom:16px;'>
            Cognitive & Educational Impacts of<br>Generative AI Usage Among University Students
        </div>
        <div style='font-size:13px; color:#94b4cc; margin-bottom:20px;'>
            <strong style='color:white;'>MSc Statistics · Team 4</strong><br>
            Vaishali Sharma &nbsp;·&nbsp; Ashish Vaghela &nbsp;·&nbsp; Raiwant Kumar &nbsp;·&nbsp; Rohan Shukla<br>
            <span style='font-size:12px;'>Guided by: Prof. Murlidharan Kunnumal</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
