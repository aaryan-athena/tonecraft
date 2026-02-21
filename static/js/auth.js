// ===== AUTH STATE MANAGEMENT =====

/**
 * Returns a promise that resolves with the current user (or null).
 * Useful for one-time checks on page load.
 */
function getCurrentUser() {
    return new Promise((resolve) => {
        const unsubscribe = auth.onAuthStateChanged((user) => {
            unsubscribe();
            resolve(user);
        });
    });
}

/**
 * Auth state observer that runs on every page.
 * Updates the navbar UI based on login state.
 * If `requireAuth` is true, redirects to /auth.html when not logged in.
 */
function initAuthStateListener(requireAuth) {
    auth.onAuthStateChanged((user) => {
        updateNavbarUI(user);

        if (requireAuth && !user) {
            var currentPage = window.location.pathname;
            window.location.href = '/auth.html?redirect=' + encodeURIComponent(currentPage);
            return;
        }

        if (requireAuth && user) {
            // Show the protected content
            var protectedContent = document.getElementById('protected-content');
            if (protectedContent) {
                protectedContent.classList.remove('hidden');
            }
            // Hide the loading placeholder
            var authLoading = document.getElementById('auth-loading');
            if (authLoading) {
                authLoading.classList.add('hidden');
            }
            // Ensure user document exists in Firestore
            ensureUserDocument(user);
        }
    });
}

// ===== NAVBAR UI =====

/**
 * Injects user info or sign-in button into the navbar.
 * Looks for an element with id="auth-nav-slot" in each page.
 */
function updateNavbarUI(user) {
    var slot = document.getElementById('auth-nav-slot');
    if (!slot) return;

    if (user) {
        var initial = (user.displayName || user.email || '?')[0].toUpperCase();
        var displayName = user.displayName || user.email.split('@')[0];
        slot.innerHTML =
            '<div class="flex items-center space-x-2 md:space-x-3">' +
                '<div class="w-8 h-8 bg-gradient-to-br from-orange-500 to-amber-600 rounded-full flex items-center justify-center text-sm font-bold">' +
                    initial +
                '</div>' +
                '<span class="hidden md:inline text-sm text-gray-300 max-w-[120px] truncate">' +
                    displayName +
                '</span>' +
                '<button onclick="signOutUser()" class="px-3 py-1.5 text-xs bg-white/10 hover:bg-white/20 rounded-lg transition-all text-gray-300">' +
                    'Sign Out' +
                '</button>' +
            '</div>';
    } else {
        slot.innerHTML =
            '<a href="/auth.html" class="px-4 py-2 bg-gradient-to-r from-orange-500 to-amber-600 hover:from-orange-600 hover:to-amber-700 rounded-lg font-semibold text-xs md:text-sm transition-all">' +
                'Sign In' +
            '</a>';
    }
}

/**
 * Signs the user out and redirects to the landing page.
 */
function signOutUser() {
    auth.signOut().then(function() {
        window.location.href = '/';
    });
}

// ===== FIRESTORE USAGE TRACKING =====

/**
 * Ensures the user document exists in Firestore.
 * Called after signup or first sign-in.
 */
function ensureUserDocument(user) {
    var userRef = db.collection('users').doc(user.uid);
    userRef.get().then(function(doc) {
        if (!doc.exists) {
            userRef.set({
                email: user.email,
                displayName: user.displayName || null,
                createdAt: firebase.firestore.FieldValue.serverTimestamp(),
                tonecraftUses: 0,
                stemflowUses: 0,
                lastUsed: firebase.firestore.FieldValue.serverTimestamp()
            });
        }
    }).catch(function(error) {
        console.error('Failed to ensure user document:', error);
    });
}

/**
 * Increments the usage count for a specific feature.
 * @param {string} feature - "tonecraft" or "stemflow"
 */
function incrementUsageCount(feature) {
    var user = auth.currentUser;
    if (!user) return;

    var field = feature === 'tonecraft' ? 'tonecraftUses' : 'stemflowUses';
    var userRef = db.collection('users').doc(user.uid);
    var updateData = {
        lastUsed: firebase.firestore.FieldValue.serverTimestamp()
    };
    updateData[field] = firebase.firestore.FieldValue.increment(1);

    userRef.update(updateData).catch(function(error) {
        console.error('Failed to increment usage count:', error);
    });
}

/**
 * Fetches the current user's usage counts.
 * Returns a promise that resolves with { tonecraftUses, stemflowUses } or null.
 */
function getUserUsageCounts() {
    var user = auth.currentUser;
    if (!user) return Promise.resolve(null);

    return db.collection('users').doc(user.uid).get().then(function(doc) {
        if (doc.exists) {
            var data = doc.data();
            return {
                tonecraftUses: data.tonecraftUses || 0,
                stemflowUses: data.stemflowUses || 0
            };
        }
        return null;
    }).catch(function(error) {
        console.error('Failed to fetch usage counts:', error);
        return null;
    });
}

// ===== ANALYTICS HELPERS =====

/**
 * Logs a page view event with the page name.
 */
function logPageView(pageName) {
    analytics.logEvent('page_view', { page_name: pageName });
}

/**
 * Logs a feature usage event.
 */
function logFeatureUsed(featureName, details) {
    var eventData = { feature: featureName };
    if (details) {
        for (var key in details) {
            if (details.hasOwnProperty(key)) {
                eventData[key] = details[key];
            }
        }
    }
    analytics.logEvent('feature_used', eventData);
}
