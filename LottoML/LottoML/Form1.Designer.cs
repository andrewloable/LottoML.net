namespace LottoML
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.chkHeader = new MaterialSkin.Controls.MaterialCheckBox();
            this.btnGenerateModel = new MaterialSkin.Controls.MaterialRaisedButton();
            this.chkSort = new MaterialSkin.Controls.MaterialCheckBox();
            this.btnBrowseCSV = new MaterialSkin.Controls.MaterialFlatButton();
            this.txtCSV = new MaterialSkin.Controls.MaterialSingleLineTextField();
            this.materialLabel1 = new MaterialSkin.Controls.MaterialLabel();
            this.txtLogs = new System.Windows.Forms.TextBox();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox1.Controls.Add(this.chkHeader);
            this.groupBox1.Controls.Add(this.btnGenerateModel);
            this.groupBox1.Controls.Add(this.chkSort);
            this.groupBox1.Controls.Add(this.btnBrowseCSV);
            this.groupBox1.Controls.Add(this.txtCSV);
            this.groupBox1.Controls.Add(this.materialLabel1);
            this.groupBox1.Location = new System.Drawing.Point(12, 76);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(776, 90);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Step 1";
            // 
            // chkHeader
            // 
            this.chkHeader.AutoSize = true;
            this.chkHeader.Depth = 0;
            this.chkHeader.Font = new System.Drawing.Font("Roboto", 10F);
            this.chkHeader.Location = new System.Drawing.Point(274, 49);
            this.chkHeader.Margin = new System.Windows.Forms.Padding(0);
            this.chkHeader.MouseLocation = new System.Drawing.Point(-1, -1);
            this.chkHeader.MouseState = MaterialSkin.MouseState.HOVER;
            this.chkHeader.Name = "chkHeader";
            this.chkHeader.Ripple = true;
            this.chkHeader.Size = new System.Drawing.Size(135, 30);
            this.chkHeader.TabIndex = 5;
            this.chkHeader.Text = "First Row Header";
            this.chkHeader.UseVisualStyleBackColor = true;
            // 
            // btnGenerateModel
            // 
            this.btnGenerateModel.AutoSize = true;
            this.btnGenerateModel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnGenerateModel.Depth = 0;
            this.btnGenerateModel.Icon = null;
            this.btnGenerateModel.Location = new System.Drawing.Point(423, 45);
            this.btnGenerateModel.MouseState = MaterialSkin.MouseState.HOVER;
            this.btnGenerateModel.Name = "btnGenerateModel";
            this.btnGenerateModel.Primary = true;
            this.btnGenerateModel.Size = new System.Drawing.Size(145, 36);
            this.btnGenerateModel.TabIndex = 4;
            this.btnGenerateModel.Text = "Start Prediction";
            this.btnGenerateModel.UseVisualStyleBackColor = true;
            this.btnGenerateModel.Click += new System.EventHandler(this.BtnGenerateModel_Click);
            // 
            // chkSort
            // 
            this.chkSort.AutoSize = true;
            this.chkSort.Checked = true;
            this.chkSort.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkSort.Depth = 0;
            this.chkSort.Font = new System.Drawing.Font("Roboto", 10F);
            this.chkSort.Location = new System.Drawing.Point(102, 49);
            this.chkSort.Margin = new System.Windows.Forms.Padding(0);
            this.chkSort.MouseLocation = new System.Drawing.Point(-1, -1);
            this.chkSort.MouseState = MaterialSkin.MouseState.HOVER;
            this.chkSort.Name = "chkSort";
            this.chkSort.Ripple = true;
            this.chkSort.Size = new System.Drawing.Size(162, 30);
            this.chkSort.TabIndex = 3;
            this.chkSort.Text = "Sort Result Sequence";
            this.chkSort.UseVisualStyleBackColor = true;
            this.chkSort.CheckedChanged += new System.EventHandler(this.MaterialCheckBox1_CheckedChanged);
            // 
            // btnBrowseCSV
            // 
            this.btnBrowseCSV.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnBrowseCSV.AutoSize = true;
            this.btnBrowseCSV.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnBrowseCSV.Depth = 0;
            this.btnBrowseCSV.Icon = null;
            this.btnBrowseCSV.Location = new System.Drawing.Point(693, 8);
            this.btnBrowseCSV.Margin = new System.Windows.Forms.Padding(4, 6, 4, 6);
            this.btnBrowseCSV.MouseState = MaterialSkin.MouseState.HOVER;
            this.btnBrowseCSV.Name = "btnBrowseCSV";
            this.btnBrowseCSV.Primary = false;
            this.btnBrowseCSV.Size = new System.Drawing.Size(76, 36);
            this.btnBrowseCSV.TabIndex = 2;
            this.btnBrowseCSV.Text = "Browse";
            this.btnBrowseCSV.UseVisualStyleBackColor = true;
            this.btnBrowseCSV.Click += new System.EventHandler(this.BtnBrowseCSV_Click);
            // 
            // txtCSV
            // 
            this.txtCSV.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtCSV.Depth = 0;
            this.txtCSV.Hint = "";
            this.txtCSV.Location = new System.Drawing.Point(102, 16);
            this.txtCSV.MaxLength = 32767;
            this.txtCSV.MouseState = MaterialSkin.MouseState.HOVER;
            this.txtCSV.Name = "txtCSV";
            this.txtCSV.PasswordChar = '\0';
            this.txtCSV.SelectedText = "";
            this.txtCSV.SelectionLength = 0;
            this.txtCSV.SelectionStart = 0;
            this.txtCSV.Size = new System.Drawing.Size(584, 23);
            this.txtCSV.TabIndex = 1;
            this.txtCSV.TabStop = false;
            this.txtCSV.UseSystemPasswordChar = false;
            // 
            // materialLabel1
            // 
            this.materialLabel1.AutoSize = true;
            this.materialLabel1.Depth = 0;
            this.materialLabel1.Font = new System.Drawing.Font("Roboto", 11F);
            this.materialLabel1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(222)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))));
            this.materialLabel1.Location = new System.Drawing.Point(6, 16);
            this.materialLabel1.MouseState = MaterialSkin.MouseState.HOVER;
            this.materialLabel1.Name = "materialLabel1";
            this.materialLabel1.Size = new System.Drawing.Size(83, 19);
            this.materialLabel1.TabIndex = 0;
            this.materialLabel1.Text = "CSV Input :";
            // 
            // txtLogs
            // 
            this.txtLogs.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txtLogs.BackColor = System.Drawing.Color.White;
            this.txtLogs.Location = new System.Drawing.Point(12, 172);
            this.txtLogs.Multiline = true;
            this.txtLogs.Name = "txtLogs";
            this.txtLogs.ReadOnly = true;
            this.txtLogs.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.txtLogs.Size = new System.Drawing.Size(776, 338);
            this.txtLogs.TabIndex = 2;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 522);
            this.Controls.Add(this.txtLogs);
            this.Controls.Add(this.groupBox1);
            this.Name = "Form1";
            this.Text = "Lotto ML";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private MaterialSkin.Controls.MaterialCheckBox chkSort;
        private MaterialSkin.Controls.MaterialFlatButton btnBrowseCSV;
        private MaterialSkin.Controls.MaterialSingleLineTextField txtCSV;
        private MaterialSkin.Controls.MaterialLabel materialLabel1;
        private MaterialSkin.Controls.MaterialRaisedButton btnGenerateModel;
        private System.Windows.Forms.TextBox txtLogs;
        private MaterialSkin.Controls.MaterialCheckBox chkHeader;
    }
}

