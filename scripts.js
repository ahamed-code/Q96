// Hardcoded login credentials
const correctEmail = "hitmanbasheer@gmail.com";
const correctPassword = "1432";

let isLoggedIn = false;

// Handle login functionality
function login() {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    if (email === correctEmail && password === correctPassword) {
        isLoggedIn = true;
        document.getElementById("loginPage").style.display = "none";
        document.getElementById("mainPage").style.display = "block";
    } else {
        document.getElementById("loginError").innerText = "Invalid email or password. Please try again.";
    }
}

// Show specific page
function showPage(page) {
    const pages = ['home', 'sip', 'dreams', 'tracker', 'portfolio'];
    pages.forEach(p => {
        document.getElementById(p).classList.add('hidden');
    });
    document.getElementById(page).classList.remove('hidden');
}

// Handle SIP Recommendation
function recommendSIP() {
    const targetAmount = document.getElementById("targetAmount").value;
    const monthlyInvestment = document.getElementById("monthlyInvestment").value;

    if (targetAmount && monthlyInvestment) {
        document.getElementById("sipRecommendation").innerHTML = `
            You have a target of ₹${targetAmount} and plan to invest ₹${monthlyInvestment} per month. 
            <br><br>Recommended Funds: 
            <ul>
                <li><a href="https://www.moneycontrol.com/mutual-funds/nifty-50">Nifty 50</a></li>
                <li><a href="https://www.moneycontrol.com/mutual-funds/sensex">Sensex</a></li>
                <li><a href="https://www.moneycontrol.com/mutual-funds/nifty-next-50">Nifty Next 50</a></li>
            </ul>
        `;
    }
}

// Handle Dream Recommendations
function recommendGoal() {
    const goal = document.getElementById("goalSelect").value;
    let recommendation = "";

    if (goal === "car") {
        recommendation = "To buy a car, we recommend investing in Nifty 50 index fund.";
    } else if (goal === "house") {
        recommendation = "To buy a house, we recommend investing in Sensex index fund.";
    } else if (goal === "gold") {
        recommendation = "To buy bulk gold, we recommend investing in Nifty Midcap 100 index fund.";
    }

    document.getElementById("dreamRecommendation").innerText = recommendation;
}

// Handle Investment Tracker functionality
function addInvestment() {
    const assetType = document.getElementById("assetType").value;
    const investmentAmount = document.getElementById("investmentAmount").value;

    if (investmentAmount) {
        document.getElementById("investmentConfirmation").innerHTML = `
            You have invested ₹${investmentAmount} in ${assetType}.
            <br>We predict a growth of 10% for this investment.
        `;
    }
}

// Handle Portfolio
let portfolio = [];

function showPortfolio() {
    const portfolioHTML = portfolio.map(item => 
        `<li>${item.asset}: ₹${item.amount} (Growth: ${item.growth})</li>`
    ).join("");

    document.getElementById("portfolioDetails").innerHTML = `<ul>${portfolioHTML}</ul>`;
}

// Logout functionality
function logout() {
    isLoggedIn = false;
    document.getElementById("loginPage").style.display = "block";
    document.getElementById("mainPage").style.display = "none";
}
